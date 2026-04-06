from __future__ import annotations

import json
from typing import Any

import torch
import torch.nn.functional as F
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict

from rl.bc_model import BCPolicyMLP
from rl.feature_encoder import ACTION_SET, action_mask_slice, action_name_to_index
from rl.feature_encoder import observation_dim


_PATCHED = False
_ORIG_PREPARE_BATCH = None
_ORIG_INIT = None
_ORIG_CALC_LOSSES = None
_ORIG_RECORD_SUMMARIES = None
_ORIG_NORMALIZE_OBS = None
_ORIG_SHARED_FORWARD_TAIL = None
_ORIG_SEPARATE_FORWARD_TAIL = None


def _row_replay_flags(row: dict) -> dict[str, float]:
    weak_actions = {"south", "west", "search"}
    behavior_action = row.get("behavior_action")
    teacher_action = row.get("teacher_action", row.get("action"))
    repeated_state_count = int(row.get("repeated_state_count", 0) or 0)
    repeated_action_count = int(row.get("repeated_action_count", 0) or 0)
    reward = float(row.get("reward", 0.0) or 0.0)
    done = bool(row.get("done", False))
    return {
        "is_disagreement_candidate": 1.0 if behavior_action is not None and behavior_action != teacher_action else 0.0,
        "is_weak_action": 1.0 if teacher_action in weak_actions else 0.0,
        "is_loop_risk": 1.0 if repeated_state_count > 0 or repeated_action_count > 0 else 0.0,
        "is_failure_slice": 1.0 if done and reward < 0.0 else 0.0,
        "teacher_action_index": float(action_name_to_index(teacher_action or "wait")),
    }


def _replay_priority_weights(
    rows: list[dict],
    source_mode: str,
    priority_power: float,
    action_boosts: dict[int, float] | None = None,
) -> torch.Tensor:
    if source_mode not in {"uniform", "weak_action", "disagreement", "mixed"}:
        raise ValueError(f"Unsupported teacher replay source mode: {source_mode}")

    base = torch.ones(len(rows), dtype=torch.float32)
    action_boosts = action_boosts or {}
    if source_mode == "uniform" and not action_boosts:
        return base

    weights = []
    for row in rows:
        flags = _row_replay_flags(row)
        disagreement = flags["is_disagreement_candidate"]
        weak_action = flags["is_weak_action"]
        loop_risk = flags["is_loop_risk"]
        failure_slice = flags["is_failure_slice"]
        if source_mode == "uniform":
            weight = 1.0
        elif source_mode == "disagreement":
            weight = 1.0 + disagreement + 0.5 * loop_risk + 0.5 * failure_slice
        elif source_mode == "weak_action":
            weight = 1.0 + weak_action + 0.5 * disagreement
        else:  # mixed
            weight = 1.0 + disagreement + weak_action + 0.5 * loop_risk + 0.5 * failure_slice
        action_idx = int(flags["teacher_action_index"])
        weight *= float(action_boosts.get(action_idx, 1.0))
        weights.append(weight)

    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    priority_power = max(0.0, float(priority_power))
    if priority_power != 1.0:
        weight_tensor = weight_tensor.pow(priority_power)
    if float(weight_tensor.sum().item()) <= 0.0:
        return base
    return weight_tensor


def _load_teacher_replay_tensors(
    path: str,
    device: torch.device,
    source_mode: str,
    priority_power: float,
    action_boosts: dict[int, float] | None = None,
) -> dict[str, torch.Tensor]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No teacher replay rows found in {path}")
    input_dims = {len(row.get("feature_vector", [])) for row in rows}
    if len(input_dims) != 1:
        raise ValueError(f"Mixed feature dimensions in teacher replay rows: {sorted(input_dims)}")
    features = torch.tensor([row["feature_vector"] for row in rows], dtype=torch.float32, device=device)
    actions = torch.tensor([ACTION_SET.index(row["action"]) for row in rows], dtype=torch.long, device=device)
    allowed_masks = torch.tensor(
        [
            [1.0 if name in row.get("allowed_actions", ACTION_SET) else 0.0 for name in ACTION_SET]
            for row in rows
        ],
        dtype=torch.float32,
        device=device,
    )
    flags = [_row_replay_flags(row) for row in rows]
    weights = _replay_priority_weights(
        rows,
        source_mode=source_mode,
        priority_power=priority_power,
        action_boosts=action_boosts,
    ).to(device)
    return {
        "features": features,
        "actions": actions,
        "allowed_masks": allowed_masks,
        "weights": weights,
        "disagreement_flags": torch.tensor([flag["is_disagreement_candidate"] for flag in flags], dtype=torch.float32, device=device),
        "weak_action_flags": torch.tensor([flag["is_weak_action"] for flag in flags], dtype=torch.float32, device=device),
        "loop_risk_flags": torch.tensor([flag["is_loop_risk"] for flag in flags], dtype=torch.float32, device=device),
        "failure_flags": torch.tensor([flag["is_failure_slice"] for flag in flags], dtype=torch.float32, device=device),
    }


def _forward_replay_action_logits(
    actor_critic: Any,
    replay_features: torch.Tensor,
    replay_allowed_masks: torch.Tensor,
) -> torch.Tensor:
    replay_core = actor_critic.forward_head({"obs": replay_features})["x"]
    replay_core = actor_critic.forward_core(replay_core, None, values_only=False)["x"]
    if hasattr(actor_critic, "decoder"):
        replay_decoder = actor_critic.decoder(replay_core)
    else:
        core_outputs = replay_core.chunk(len(actor_critic.cores), dim=1)
        replay_decoder = actor_critic.actor_decoder(core_outputs[0])
    replay_logits, _ = actor_critic.action_parameterization(replay_decoder)
    return replay_logits.masked_fill(replay_allowed_masks <= 0, -1e9)


def _teacher_enabled(cfg: Any) -> bool:
    return bool(_parse_teacher_bc_paths(getattr(cfg, "teacher_bc_path", None))) and float(getattr(cfg, "teacher_loss_coef", 0.0) or 0.0) > 0.0


def _resolve_teacher_prior_bc_paths(cfg: Any) -> list[str]:
    prior_raw = getattr(cfg, "teacher_prior_bc_path", None)
    if prior_raw:
        return _parse_teacher_bc_paths(prior_raw)
    return _parse_teacher_bc_paths(getattr(cfg, "teacher_bc_path", None))


def _teacher_policy_prior_enabled(cfg: Any) -> bool:
    return bool(_resolve_teacher_prior_bc_paths(cfg)) and (
        float(getattr(cfg, "teacher_policy_logit_residual_scale", 1.0) or 1.0) != 1.0
        or
        float(getattr(cfg, "teacher_policy_blend_coef", 0.0) or 0.0) > 0.0
        or float(getattr(cfg, "teacher_policy_fallback_confidence", 0.0) or 0.0) > 0.0
        or float(getattr(cfg, "teacher_policy_disagreement_margin", 0.0) or 0.0) > 0.0
    )


def _teacher_model_enabled(cfg: Any) -> bool:
    return _teacher_enabled(cfg) or _teacher_policy_prior_enabled(cfg)


def _parse_teacher_bc_paths(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def _scheduled_teacher_coef(self) -> float:
    start = float(getattr(self, "teacher_loss_coef", 0.0) or 0.0)
    final = float(getattr(self, "teacher_loss_final_coef", 0.0) or 0.0)
    warmup = int(getattr(self, "teacher_loss_warmup_env_steps", 0) or 0)
    decay = int(getattr(self, "teacher_loss_decay_env_steps", 0) or 0)
    if decay <= 0:
        return start
    env_steps = int(getattr(self, "env_steps", 0) or 0)
    if env_steps <= warmup:
        return start
    progress = min(1.0, max(0.0, (env_steps - warmup) / decay))
    return start + (final - start) * progress


def _scheduled_teacher_replay_coef(self) -> float:
    start = float(getattr(self, "teacher_replay_coef", 0.0) or 0.0)
    final = float(getattr(self, "teacher_replay_final_coef", 0.0) or 0.0)
    warmup = int(getattr(self, "teacher_replay_warmup_env_steps", 0) or 0)
    decay = int(getattr(self, "teacher_replay_decay_env_steps", 0) or 0)
    if decay <= 0:
        return start
    env_steps = int(getattr(self, "env_steps", 0) or 0)
    if env_steps <= warmup:
        return start
    progress = min(1.0, max(0.0, (env_steps - warmup) / decay))
    return start + (final - start) * progress


def _scheduled_actor_loss_scale(self) -> float:
    start_raw = getattr(self, "actor_loss_scale", None)
    start = 1.0 if start_raw is None else float(start_raw)
    final_raw = getattr(self, "actor_loss_final_scale", None)
    final = start if final_raw is None else float(final_raw)
    warmup = int(getattr(self, "actor_loss_warmup_env_steps", 0) or 0)
    decay = int(getattr(self, "actor_loss_decay_env_steps", 0) or 0)
    if decay <= 0:
        return start
    env_steps = int(getattr(self, "env_steps", 0) or 0)
    if env_steps <= warmup:
        return start
    progress = min(1.0, max(0.0, (env_steps - warmup) / decay))
    return start + (final - start) * progress


def _anchor_named_parameters(self):
    actor_critic = getattr(self, "actor_critic", None)
    if actor_critic is None:
        return {}
    named: dict[str, torch.Tensor] = {}
    if hasattr(actor_critic, "actor_encoder"):
        mlp_head = actor_critic.actor_encoder.encoders["obs"].mlp_head
    else:
        mlp_head = actor_critic.encoder.encoders["obs"].mlp_head
    linear_idx = 0
    for module in mlp_head:
        if isinstance(module, torch.nn.Linear):
            named[f"actor_encoder_{linear_idx}_weight"] = module.weight
            named[f"actor_encoder_{linear_idx}_bias"] = module.bias
            linear_idx += 1
    named["policy_weight"] = actor_critic.action_parameterization.distribution_linear.weight
    named["policy_bias"] = actor_critic.action_parameterization.distribution_linear.bias
    return named


def _parse_teacher_action_boosts(raw: str) -> dict[int, float]:
    boosts: dict[int, float] = {}
    if not raw:
        return boosts

    action_names = {
        "north": 0,
        "south": 1,
        "east": 2,
        "west": 3,
        "wait": 4,
        "search": 5,
        "pickup": 6,
        "open": 7,
        "kick": 8,
        "apply": 9,
        "eat": 10,
        "quaff": 11,
        "zap": 12,
    }
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid teacher_action_boost entry: {item!r}")
        name, value = item.split("=", 1)
        key = name.strip().lower()
        if key not in action_names:
            raise ValueError(f"Unknown teacher action in boost spec: {name!r}")
        boosts[action_names[key]] = float(value.strip())
    return boosts


def _action_mask_from_raw_obs(raw_obs: torch.Tensor) -> torch.Tensor:
    mask = raw_obs[..., action_mask_slice()]
    if mask.shape[-1] != len(ACTION_SET):
        raise ValueError(f"Expected action mask width {len(ACTION_SET)}, got {mask.shape[-1]}")
    return mask


def _mask_logits_with_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(action_mask <= 0, -1e9)


def _invalid_preference_fraction(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    preferred_actions = torch.argmax(logits, dim=-1)
    invalid = action_mask.gather(1, preferred_actions.unsqueeze(1)).squeeze(1) <= 0
    return invalid.float().mean()


def _load_bc_teacher_model(teacher_path: str, device: torch.device) -> BCPolicyMLP:
    payload = torch.load(teacher_path, map_location=device)
    metadata = payload.get("metadata", {})
    input_dim = int(metadata.get("input_dim", observation_dim(metadata.get("observation_version", "v1"))))
    hidden_size = int(metadata.get("hidden_size", 256))
    num_layers = int(metadata.get("num_layers", 2))
    teacher = BCPolicyMLP(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers)
    teacher.load_state_dict(payload["state_dict"])
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _teacher_policy_blend(student_probs: torch.Tensor, teacher_probs: torch.Tensor, blend_coef: float) -> torch.Tensor:
    blend = float(min(max(blend_coef, 0.0), 1.0))
    if blend <= 0.0:
        return student_probs
    return (1.0 - blend) * student_probs + blend * teacher_probs


def _teacher_policy_logit_residual(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    residual_scale: float,
) -> torch.Tensor:
    scale = float(min(max(residual_scale, 0.0), 1.0))
    if scale >= 1.0:
        return student_logits
    if scale <= 0.0:
        return teacher_logits
    return teacher_logits + scale * (student_logits - teacher_logits)


def _teacher_policy_fallback_details(
    student_probs: torch.Tensor,
    confidence_threshold: float,
    teacher_probs: torch.Tensor | None = None,
    disagreement_margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    threshold = float(confidence_threshold)
    confidence_mask = torch.zeros(student_probs.shape[0], dtype=torch.bool, device=student_probs.device)
    if threshold > 0.0:
        confidence_mask = student_probs.max(dim=-1).values < threshold

    disagreement_mask = torch.zeros_like(confidence_mask)
    weak_override_mask = torch.zeros_like(confidence_mask)
    margin = float(disagreement_margin)
    if teacher_probs is not None:
        student_actions = torch.argmax(student_probs, dim=-1)
        teacher_actions = torch.argmax(teacher_probs, dim=-1)
        disagreement_mask = student_actions != teacher_actions
        if margin > 0.0:
            student_top_probs = student_probs.gather(-1, student_actions.unsqueeze(-1)).squeeze(-1)
            teacher_action_probs = student_probs.gather(-1, teacher_actions.unsqueeze(-1)).squeeze(-1)
            weak_override_mask = disagreement_mask & ((student_top_probs - teacher_action_probs) < margin)

    return confidence_mask | weak_override_mask, disagreement_mask, weak_override_mask


def _teacher_policy_fallback_mask(
    student_probs: torch.Tensor,
    confidence_threshold: float,
    teacher_probs: torch.Tensor | None = None,
    disagreement_margin: float = 0.0,
) -> torch.Tensor:
    fallback_mask, _, _ = _teacher_policy_fallback_details(
        student_probs,
        confidence_threshold,
        teacher_probs=teacher_probs,
        disagreement_margin=disagreement_margin,
    )
    return fallback_mask


def _ensure_teacher_prior_models(module: Any, device: torch.device) -> None:
    if getattr(module, "_teacher_prior_initialized", False):
        return
    cfg = getattr(module, "cfg", None)
    module._teacher_prior_initialized = True
    module._teacher_prior_policies = []
    module._teacher_policy_logit_residual_scale = float(
        getattr(cfg, "teacher_policy_logit_residual_scale", 1.0) or 1.0
    )
    module._teacher_policy_blend_coef = float(getattr(cfg, "teacher_policy_blend_coef", 0.0) or 0.0)
    module._teacher_policy_fallback_confidence = float(
        getattr(cfg, "teacher_policy_fallback_confidence", 0.0) or 0.0
    )
    module._teacher_policy_disagreement_margin = float(
        getattr(cfg, "teacher_policy_disagreement_margin", 0.0) or 0.0
    )
    if cfg is None or not _teacher_policy_prior_enabled(cfg):
        return
    if bool(getattr(cfg, "normalize_input", False)):
        raise ValueError("Teacher policy prior requires normalize_input=False")
    teacher_paths = _resolve_teacher_prior_bc_paths(cfg)
    module._teacher_prior_policies = [_load_bc_teacher_model(path, device) for path in teacher_paths]


def _apply_teacher_policy_prior(module: Any, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_teacher_prior_models(module, action_logits.device)
    teacher_policies = getattr(module, "_teacher_prior_policies", [])
    if not teacher_policies:
        zero = action_logits.new_zeros(())
        return action_logits, zero, zero
    raw_obs = getattr(module, "_teacher_prior_raw_obs", None)
    if raw_obs is None:
        zero = action_logits.new_zeros(())
        return action_logits, zero, zero
    raw_obs = raw_obs.to(action_logits.device)
    allowed_action_mask = _action_mask_from_raw_obs(raw_obs)
    with torch.no_grad():
        teacher_logits_raw = torch.stack([teacher(raw_obs.float()) for teacher in teacher_policies], dim=0).mean(dim=0)
    student_logits = _mask_logits_with_action_mask(action_logits, allowed_action_mask)
    teacher_logits = _mask_logits_with_action_mask(teacher_logits_raw, allowed_action_mask)
    residual_logits = _teacher_policy_logit_residual(
        student_logits,
        teacher_logits,
        getattr(module, "_teacher_policy_logit_residual_scale", 1.0),
    )
    student_probs = torch.softmax(residual_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    blended_probs = _teacher_policy_blend(
        student_probs,
        teacher_probs,
        getattr(module, "_teacher_policy_blend_coef", 0.0),
    )
    fallback_mask, _, _ = _teacher_policy_fallback_details(
        student_probs,
        getattr(module, "_teacher_policy_fallback_confidence", 0.0),
        teacher_probs=teacher_probs,
        disagreement_margin=getattr(module, "_teacher_policy_disagreement_margin", 0.0),
    )
    if fallback_mask.any():
        blended_probs = blended_probs.clone()
        blended_probs[fallback_mask] = teacher_probs[fallback_mask]
    blended_logits = torch.log(blended_probs.clamp_min(1e-8))
    prior_applied_fraction = action_logits.new_full((), 1.0)
    fallback_fraction = fallback_mask.float().mean() if fallback_mask.numel() > 0 else action_logits.new_zeros(())
    return blended_logits, prior_applied_fraction, fallback_fraction


def patch_sample_factory_teacher_reg() -> None:
    global _PATCHED, _ORIG_PREPARE_BATCH, _ORIG_INIT, _ORIG_CALC_LOSSES, _ORIG_RECORD_SUMMARIES
    global _ORIG_NORMALIZE_OBS, _ORIG_SHARED_FORWARD_TAIL, _ORIG_SEPARATE_FORWARD_TAIL
    if _PATCHED:
        return

    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.torch_utils import masked_select
    from sample_factory.model.actor_critic import ActorCritic, ActorCriticSeparateWeights, ActorCriticSharedWeights

    _ORIG_PREPARE_BATCH = Learner._prepare_batch
    _ORIG_INIT = Learner.init
    _ORIG_CALC_LOSSES = Learner._calculate_losses
    _ORIG_RECORD_SUMMARIES = Learner._record_summaries
    _ORIG_NORMALIZE_OBS = ActorCritic.normalize_obs
    _ORIG_SHARED_FORWARD_TAIL = ActorCriticSharedWeights.forward_tail
    _ORIG_SEPARATE_FORWARD_TAIL = ActorCriticSeparateWeights.forward_tail

    def _prepare_batch_with_raw_obs(self, batch):
        raw_obs = None
        if _teacher_model_enabled(self.cfg):
            raw_obs = batch["obs"]["obs"][:, :-1].reshape(-1, batch["obs"]["obs"].shape[-1]).clone()
        buff, experience_size, num_invalids = _ORIG_PREPARE_BATCH(self, batch)
        if raw_obs is not None:
            buff["raw_obs"] = raw_obs
        return buff, experience_size, num_invalids

    def _init_with_teacher(self):
        model_init = _ORIG_INIT(self)
        self.teacher_policy = None
        self.teacher_policies = []
        self.teacher_replay = None
        self.teacher_loss_enabled = False
        self.teacher_loss_coef = float(getattr(self.cfg, "teacher_loss_coef", 0.0) or 0.0)
        self.teacher_loss_type = str(getattr(self.cfg, "teacher_loss_type", "ce") or "ce")
        self.teacher_bc_path = getattr(self.cfg, "teacher_bc_path", None)
        self.teacher_action_boosts = _parse_teacher_action_boosts(
            str(getattr(self.cfg, "teacher_action_boosts", "") or "")
        )
        self.teacher_loss_final_coef = float(getattr(self.cfg, "teacher_loss_final_coef", 0.0) or 0.0)
        self.teacher_loss_warmup_env_steps = int(getattr(self.cfg, "teacher_loss_warmup_env_steps", 0) or 0)
        self.teacher_loss_decay_env_steps = int(getattr(self.cfg, "teacher_loss_decay_env_steps", 0) or 0)
        self.teacher_replay_trace_input = getattr(self.cfg, "teacher_replay_trace_input", None)
        self.teacher_replay_coef = float(getattr(self.cfg, "teacher_replay_coef", 0.0) or 0.0)
        self.teacher_replay_final_coef = float(getattr(self.cfg, "teacher_replay_final_coef", 0.0) or 0.0)
        self.teacher_replay_warmup_env_steps = int(getattr(self.cfg, "teacher_replay_warmup_env_steps", 0) or 0)
        self.teacher_replay_decay_env_steps = int(getattr(self.cfg, "teacher_replay_decay_env_steps", 0) or 0)
        self.teacher_replay_batch_size = int(getattr(self.cfg, "teacher_replay_batch_size", 128) or 128)
        self.teacher_replay_priority_power = float(getattr(self.cfg, "teacher_replay_priority_power", 1.0) or 1.0)
        self.teacher_replay_source_mode = str(getattr(self.cfg, "teacher_replay_source_mode", "uniform") or "uniform")
        self.teacher_replay_action_boosts = _parse_teacher_action_boosts(
            str(getattr(self.cfg, "teacher_replay_action_boosts", "") or "")
        )
        self.param_anchor_coef = float(getattr(self.cfg, "param_anchor_coef", 0.0) or 0.0)
        self.actor_loss_scale = float(getattr(self.cfg, "actor_loss_scale", 1.0) or 1.0)
        self.actor_loss_final_scale = float(getattr(self.cfg, "actor_loss_final_scale", 1.0) or 1.0)
        self.actor_loss_warmup_env_steps = int(getattr(self.cfg, "actor_loss_warmup_env_steps", 0) or 0)
        self.actor_loss_decay_env_steps = int(getattr(self.cfg, "actor_loss_decay_env_steps", 0) or 0)
        self.param_anchor_tensors = {}
        if self.teacher_loss_type not in {"ce", "kl"}:
            raise ValueError(f"Unsupported teacher_loss_type: {self.teacher_loss_type}")
        self.teacher_loss_enabled = _teacher_enabled(self.cfg)
        if _teacher_model_enabled(self.cfg):
            if getattr(self.cfg, "use_rnn", False):
                raise ValueError("Teacher-regularized APPO is only supported on the non-RNN path")
            teacher_paths = _parse_teacher_bc_paths(self.teacher_bc_path)
            for teacher_path in teacher_paths:
                self.teacher_policies.append(_load_bc_teacher_model(teacher_path, self.device))
            self.teacher_policy = self.teacher_policies[0] if self.teacher_policies else None
        if self.teacher_replay_trace_input and max(self.teacher_replay_coef, self.teacher_replay_final_coef) > 0.0:
            self.teacher_replay = _load_teacher_replay_tensors(
                self.teacher_replay_trace_input,
                self.device,
                source_mode=self.teacher_replay_source_mode,
                priority_power=self.teacher_replay_priority_power,
                action_boosts=self.teacher_replay_action_boosts,
            )
        if self.param_anchor_coef > 0.0:
            self.param_anchor_tensors = {
                name: param.detach().clone().to(self.device)
                for name, param in _anchor_named_parameters(self).items()
            }
        return model_init

    def _calculate_losses_with_teacher(self, mb, num_invalids):
        result = list(_ORIG_CALC_LOSSES(self, mb, num_invalids))
        action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries = result
        teacher_loss = torch.zeros((), device=self.device)
        teacher_agreement = torch.zeros((), device=self.device)
        teacher_replay_loss = torch.zeros((), device=self.device)
        teacher_replay_coef = torch.zeros((), device=self.device)
        param_anchor_loss = torch.zeros((), device=self.device)
        teacher_invalid_preference_fraction = torch.zeros((), device=self.device)
        student_invalid_preference_fraction = torch.zeros((), device=self.device)
        teacher_policy_prior_applied_fraction = torch.zeros((), device=self.device)
        teacher_policy_fallback_fraction = torch.zeros((), device=self.device)
        teacher_policy_disagreement_fraction = torch.zeros((), device=self.device)
        teacher_policy_weak_override_fraction = torch.zeros((), device=self.device)
        actor_loss_scale = policy_loss.new_tensor(_scheduled_actor_loss_scale(self))

        if self.teacher_policies and self.teacher_loss_enabled:
            raw_obs = mb.get("raw_obs")
            if raw_obs is None:
                raise RuntimeError("Teacher regularization requires raw_obs in the minibatch")
            allowed_action_mask = _action_mask_from_raw_obs(raw_obs).to(self.device)
            with torch.no_grad():
                teacher_logits_raw = torch.stack([teacher(raw_obs.float()) for teacher in self.teacher_policies], dim=0).mean(dim=0)
                teacher_invalid_preference_fraction = _invalid_preference_fraction(teacher_logits_raw, allowed_action_mask)
                teacher_logits = _mask_logits_with_action_mask(teacher_logits_raw, allowed_action_mask)
                teacher_actions = torch.argmax(teacher_logits, dim=-1)

            student_logits_raw = action_distribution.raw_logits
            student_invalid_preference_fraction = _invalid_preference_fraction(student_logits_raw, allowed_action_mask)
            student_logits = _mask_logits_with_action_mask(student_logits_raw, allowed_action_mask)
            if self.teacher_loss_type == "ce":
                teacher_loss_all = F.cross_entropy(student_logits, teacher_actions, reduction="none")
            else:
                teacher_probs = torch.softmax(teacher_logits, dim=-1)
                teacher_loss_all = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_probs,
                    reduction="none",
                ).sum(dim=-1)

            if self.teacher_action_boosts:
                weight = torch.ones_like(teacher_loss_all)
                for action_id, multiplier in self.teacher_action_boosts.items():
                    weight = torch.where(
                        teacher_actions == action_id,
                        weight.new_full(weight.shape, float(multiplier)),
                        weight,
                    )
                teacher_loss_all = teacher_loss_all * weight

            teacher_loss = masked_select(teacher_loss_all, mb.valids, num_invalids).mean()
            teacher_loss = teacher_loss * _scheduled_teacher_coef(self)

            student_actions = torch.argmax(student_logits, dim=-1)
            agreement_all = (student_actions == teacher_actions).float()
            teacher_agreement = masked_select(agreement_all, mb.valids, num_invalids).mean()
            teacher_policy_prior_applied_fraction = policy_loss.new_tensor(
                1.0 if _teacher_policy_prior_enabled(self.cfg) else 0.0
            )
            fallback_mask, disagreement_mask, weak_override_mask = _teacher_policy_fallback_details(
                torch.softmax(student_logits, dim=-1),
                float(getattr(self.cfg, "teacher_policy_fallback_confidence", 0.0) or 0.0),
                teacher_probs=torch.softmax(teacher_logits, dim=-1),
                disagreement_margin=float(getattr(self.cfg, "teacher_policy_disagreement_margin", 0.0) or 0.0),
            )
            teacher_policy_fallback_fraction = fallback_mask.float().mean()
            teacher_policy_disagreement_fraction = disagreement_mask.float().mean()
            teacher_policy_weak_override_fraction = weak_override_mask.float().mean()

        if self.teacher_replay is not None and max(self.teacher_replay_coef, self.teacher_replay_final_coef) > 0.0:
            replay_size = self.teacher_replay["features"].shape[0]
            batch_size = min(self.teacher_replay_batch_size, replay_size)
            replay_indices = torch.multinomial(self.teacher_replay["weights"], batch_size, replacement=True)
            replay_features = self.teacher_replay["features"][replay_indices]
            replay_actions = self.teacher_replay["actions"][replay_indices]
            replay_allowed_masks = self.teacher_replay["allowed_masks"][replay_indices]
            replay_disagreement_flags = self.teacher_replay["disagreement_flags"][replay_indices]
            replay_weak_action_flags = self.teacher_replay["weak_action_flags"][replay_indices]
            replay_loop_risk_flags = self.teacher_replay["loop_risk_flags"][replay_indices]
            replay_failure_flags = self.teacher_replay["failure_flags"][replay_indices]
            student_replay_logits = _forward_replay_action_logits(
                self.actor_critic,
                replay_features,
                replay_allowed_masks,
            )
            teacher_replay_coef = teacher_replay_loss.new_tensor(_scheduled_teacher_replay_coef(self))
            teacher_replay_loss = F.cross_entropy(student_replay_logits, replay_actions) * teacher_replay_coef
            loss_summaries["teacher_replay_disagreement_fraction"] = replay_disagreement_flags.mean()
            loss_summaries["teacher_replay_weak_action_fraction"] = replay_weak_action_flags.mean()
            loss_summaries["teacher_replay_loop_risk_fraction"] = replay_loop_risk_flags.mean()
            loss_summaries["teacher_replay_failure_fraction"] = replay_failure_flags.mean()

        if self.param_anchor_coef > 0.0 and self.param_anchor_tensors:
            anchor_terms = []
            for name, param in _anchor_named_parameters(self).items():
                anchor = self.param_anchor_tensors.get(name)
                if anchor is None:
                    continue
                anchor_terms.append(F.mse_loss(param, anchor, reduction="mean"))
            if anchor_terms:
                param_anchor_loss = torch.stack(anchor_terms).mean() * self.param_anchor_coef

        loss_summaries["teacher_loss"] = teacher_loss
        loss_summaries["teacher_agreement"] = teacher_agreement
        loss_summaries["teacher_replay_loss"] = teacher_replay_loss
        loss_summaries["teacher_replay_coef"] = teacher_replay_coef
        loss_summaries["teacher_replay_priority_power"] = teacher_replay_loss.new_tensor(
            float(getattr(self, "teacher_replay_priority_power", 1.0) or 1.0)
        )
        loss_summaries["actor_loss_scale"] = actor_loss_scale
        loss_summaries["param_anchor_loss"] = param_anchor_loss
        loss_summaries["teacher_invalid_preference_fraction"] = teacher_invalid_preference_fraction
        loss_summaries["student_invalid_preference_fraction"] = student_invalid_preference_fraction
        loss_summaries["teacher_policy_prior_applied_fraction"] = teacher_policy_prior_applied_fraction
        loss_summaries["teacher_policy_fallback_fraction"] = teacher_policy_fallback_fraction
        loss_summaries["teacher_policy_disagreement_fraction"] = teacher_policy_disagreement_fraction
        loss_summaries["teacher_policy_weak_override_fraction"] = teacher_policy_weak_override_fraction
        result[1] = actor_loss_scale * policy_loss + teacher_loss + teacher_replay_loss + param_anchor_loss
        result[6] = loss_summaries
        return tuple(result)

    def _record_summaries_with_teacher(self, train_loop_vars):
        stats = _ORIG_RECORD_SUMMARIES(self, train_loop_vars)
        if hasattr(train_loop_vars, "teacher_loss"):
            stats.teacher_loss = float(train_loop_vars.teacher_loss.item())
        if hasattr(train_loop_vars, "teacher_agreement"):
            stats.teacher_agreement = float(train_loop_vars.teacher_agreement.item())
        if hasattr(train_loop_vars, "teacher_replay_loss"):
            stats.teacher_replay_loss = float(train_loop_vars.teacher_replay_loss.item())
        if hasattr(train_loop_vars, "teacher_replay_coef"):
            stats.teacher_replay_coef = float(train_loop_vars.teacher_replay_coef.item())
        if hasattr(train_loop_vars, "teacher_replay_priority_power"):
            stats.teacher_replay_priority_power = float(train_loop_vars.teacher_replay_priority_power.item())
        if hasattr(train_loop_vars, "actor_loss_scale"):
            stats.actor_loss_scale = float(train_loop_vars.actor_loss_scale.item())
        if hasattr(train_loop_vars, "teacher_replay_disagreement_fraction"):
            stats.teacher_replay_disagreement_fraction = float(train_loop_vars.teacher_replay_disagreement_fraction.item())
        if hasattr(train_loop_vars, "teacher_replay_weak_action_fraction"):
            stats.teacher_replay_weak_action_fraction = float(train_loop_vars.teacher_replay_weak_action_fraction.item())
        if hasattr(train_loop_vars, "teacher_replay_loop_risk_fraction"):
            stats.teacher_replay_loop_risk_fraction = float(train_loop_vars.teacher_replay_loop_risk_fraction.item())
        if hasattr(train_loop_vars, "teacher_replay_failure_fraction"):
            stats.teacher_replay_failure_fraction = float(train_loop_vars.teacher_replay_failure_fraction.item())
        if hasattr(train_loop_vars, "param_anchor_loss"):
            stats.param_anchor_loss = float(train_loop_vars.param_anchor_loss.item())
        if hasattr(train_loop_vars, "teacher_invalid_preference_fraction"):
            stats.teacher_invalid_preference_fraction = float(train_loop_vars.teacher_invalid_preference_fraction.item())
        if hasattr(train_loop_vars, "student_invalid_preference_fraction"):
            stats.student_invalid_preference_fraction = float(train_loop_vars.student_invalid_preference_fraction.item())
        if hasattr(train_loop_vars, "teacher_policy_prior_applied_fraction"):
            stats.teacher_policy_prior_applied_fraction = float(train_loop_vars.teacher_policy_prior_applied_fraction.item())
        if hasattr(train_loop_vars, "teacher_policy_fallback_fraction"):
            stats.teacher_policy_fallback_fraction = float(train_loop_vars.teacher_policy_fallback_fraction.item())
        if hasattr(train_loop_vars, "teacher_policy_disagreement_fraction"):
            stats.teacher_policy_disagreement_fraction = float(train_loop_vars.teacher_policy_disagreement_fraction.item())
        if hasattr(train_loop_vars, "teacher_policy_weak_override_fraction"):
            stats.teacher_policy_weak_override_fraction = float(train_loop_vars.teacher_policy_weak_override_fraction.item())
        return stats

    def _normalize_obs_with_teacher_prior(self, obs):
        if isinstance(obs, dict) and "obs" in obs:
            self._teacher_prior_raw_obs = obs["obs"].detach().clone()
        return _ORIG_NORMALIZE_OBS(self, obs)

    def _forward_tail_shared_with_teacher_prior(self, core_output, values_only: bool, sample_actions: bool):
        decoder_output = self.decoder(core_output)
        values = self.critic_linear(decoder_output).squeeze()
        result = TensorDict(values=values)
        if values_only:
            return result
        action_distribution_params, self.last_action_distribution = self.action_parameterization(decoder_output)
        action_distribution_params, teacher_prior_fraction, teacher_fallback_fraction = _apply_teacher_policy_prior(
            self, action_distribution_params
        )
        self.last_action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        result["action_logits"] = action_distribution_params
        result["teacher_policy_prior_applied_fraction"] = teacher_prior_fraction
        result["teacher_policy_fallback_fraction"] = teacher_fallback_fraction
        self._maybe_sample_actions(sample_actions, result)
        return result

    def _forward_tail_separate_with_teacher_prior(self, core_output, values_only: bool, sample_actions: bool):
        core_outputs = core_output.chunk(len(self.cores), dim=1)
        critic_decoder_output = self.critic_decoder(core_outputs[1])
        values = self.critic_linear(critic_decoder_output).squeeze()
        result = TensorDict(values=values)
        if values_only:
            return result
        actor_decoder_output = self.actor_decoder(core_outputs[0])
        action_distribution_params, self.last_action_distribution = self.action_parameterization(actor_decoder_output)
        action_distribution_params, teacher_prior_fraction, teacher_fallback_fraction = _apply_teacher_policy_prior(
            self, action_distribution_params
        )
        self.last_action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        result["action_logits"] = action_distribution_params
        result["teacher_policy_prior_applied_fraction"] = teacher_prior_fraction
        result["teacher_policy_fallback_fraction"] = teacher_fallback_fraction
        self._maybe_sample_actions(sample_actions, result)
        return result

    Learner._prepare_batch = _prepare_batch_with_raw_obs
    Learner.init = _init_with_teacher
    Learner._calculate_losses = _calculate_losses_with_teacher
    Learner._record_summaries = _record_summaries_with_teacher
    ActorCritic.normalize_obs = _normalize_obs_with_teacher_prior
    ActorCriticSharedWeights.forward_tail = _forward_tail_shared_with_teacher_prior
    ActorCriticSeparateWeights.forward_tail = _forward_tail_separate_with_teacher_prior
    _PATCHED = True
