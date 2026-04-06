from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rl.bc_model import BCPolicyMLP
from rl.feature_encoder import observation_dim


_PATCHED = False
_ORIG_PREPARE_BATCH = None
_ORIG_INIT = None
_ORIG_CALC_LOSSES = None
_ORIG_RECORD_SUMMARIES = None


def _teacher_enabled(cfg: Any) -> bool:
    return bool(getattr(cfg, "teacher_bc_path", None)) and float(getattr(cfg, "teacher_loss_coef", 0.0) or 0.0) > 0.0


def _anchor_named_parameters(self):
    actor_critic = getattr(self, "actor_critic", None)
    if actor_critic is None:
        return {}
    return {
        "encoder_0_weight": actor_critic.encoder.encoders["obs"].mlp_head[0].weight,
        "encoder_0_bias": actor_critic.encoder.encoders["obs"].mlp_head[0].bias,
        "encoder_2_weight": actor_critic.encoder.encoders["obs"].mlp_head[2].weight,
        "encoder_2_bias": actor_critic.encoder.encoders["obs"].mlp_head[2].bias,
        "policy_weight": actor_critic.action_parameterization.distribution_linear.weight,
        "policy_bias": actor_critic.action_parameterization.distribution_linear.bias,
    }


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


def patch_sample_factory_teacher_reg() -> None:
    global _PATCHED, _ORIG_PREPARE_BATCH, _ORIG_INIT, _ORIG_CALC_LOSSES, _ORIG_RECORD_SUMMARIES
    if _PATCHED:
        return

    from sample_factory.algo.learning.learner import Learner
    from sample_factory.algo.utils.torch_utils import masked_select

    _ORIG_PREPARE_BATCH = Learner._prepare_batch
    _ORIG_INIT = Learner.init
    _ORIG_CALC_LOSSES = Learner._calculate_losses
    _ORIG_RECORD_SUMMARIES = Learner._record_summaries

    def _prepare_batch_with_raw_obs(self, batch):
        raw_obs = None
        if _teacher_enabled(self.cfg):
            raw_obs = batch["obs"]["obs"][:, :-1].reshape(-1, batch["obs"]["obs"].shape[-1]).clone()
        buff, experience_size, num_invalids = _ORIG_PREPARE_BATCH(self, batch)
        if raw_obs is not None:
            buff["raw_obs"] = raw_obs
        return buff, experience_size, num_invalids

    def _init_with_teacher(self):
        model_init = _ORIG_INIT(self)
        self.teacher_policy = None
        self.teacher_loss_coef = float(getattr(self.cfg, "teacher_loss_coef", 0.0) or 0.0)
        self.teacher_loss_type = str(getattr(self.cfg, "teacher_loss_type", "ce") or "ce")
        self.teacher_bc_path = getattr(self.cfg, "teacher_bc_path", None)
        self.teacher_action_boosts = _parse_teacher_action_boosts(
            str(getattr(self.cfg, "teacher_action_boosts", "") or "")
        )
        self.param_anchor_coef = float(getattr(self.cfg, "param_anchor_coef", 0.0) or 0.0)
        self.param_anchor_tensors = {}
        if self.teacher_loss_type not in {"ce", "kl"}:
            raise ValueError(f"Unsupported teacher_loss_type: {self.teacher_loss_type}")
        if _teacher_enabled(self.cfg):
            if getattr(self.cfg, "use_rnn", False):
                raise ValueError("Teacher-regularized APPO is only supported on the non-RNN path")
            input_dim = observation_dim(getattr(self.cfg, "observation_version", "v1"))
            payload = torch.load(self.teacher_bc_path, map_location=self.device)
            metadata = payload.get("metadata", {})
            hidden_size = int(metadata.get("hidden_size", 256))
            teacher = BCPolicyMLP(input_dim=input_dim, hidden_size=hidden_size)
            teacher.load_state_dict(payload["state_dict"])
            teacher.to(self.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teacher_policy = teacher
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
        param_anchor_loss = torch.zeros((), device=self.device)

        if self.teacher_policy is not None:
            raw_obs = mb.get("raw_obs")
            if raw_obs is None:
                raise RuntimeError("Teacher regularization requires raw_obs in the minibatch")
            with torch.no_grad():
                teacher_logits = self.teacher_policy(raw_obs.float())
                teacher_actions = torch.argmax(teacher_logits, dim=-1)

            student_logits = action_distribution.raw_logits
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
            teacher_loss = teacher_loss * self.teacher_loss_coef

            student_actions = torch.argmax(student_logits, dim=-1)
            agreement_all = (student_actions == teacher_actions).float()
            teacher_agreement = masked_select(agreement_all, mb.valids, num_invalids).mean()

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
        loss_summaries["param_anchor_loss"] = param_anchor_loss
        result[1] = policy_loss + teacher_loss + param_anchor_loss
        result[6] = loss_summaries
        return tuple(result)

    def _record_summaries_with_teacher(self, train_loop_vars):
        stats = _ORIG_RECORD_SUMMARIES(self, train_loop_vars)
        if hasattr(train_loop_vars, "teacher_loss"):
            stats.teacher_loss = float(train_loop_vars.teacher_loss.item())
        if hasattr(train_loop_vars, "teacher_agreement"):
            stats.teacher_agreement = float(train_loop_vars.teacher_agreement.item())
        if hasattr(train_loop_vars, "param_anchor_loss"):
            stats.param_anchor_loss = float(train_loop_vars.param_anchor_loss.item())
        return stats

    Learner._prepare_batch = _prepare_batch_with_raw_obs
    Learner.init = _init_with_teacher
    Learner._calculate_losses = _calculate_losses_with_teacher
    Learner._record_summaries = _record_summaries_with_teacher
    _PATCHED = True
