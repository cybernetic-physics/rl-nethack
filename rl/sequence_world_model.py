from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET, SKILL_SET
from rl.io_utils import atomic_torch_save
from rl.world_model import HashTextEncoder


def _split_latent_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean, logvar = torch.chunk(tensor, 2, dim=-1)
    return mean, logvar.clamp(min=-8.0, max=8.0)


def _sample_gaussian(mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    if deterministic:
        return mean
    std = torch.exp(0.5 * logvar)
    return mean + torch.randn_like(std) * std


class SequenceWorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_size: int = 128,
        action_dim: int = len(ACTION_SET),
        skill_dim: int = len(SKILL_SET),
        text_encoder_backend: str = "none",
        text_embedding_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.text_encoder_backend = text_encoder_backend
        self.text_embedding_dim = text_embedding_dim

        self.obs_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        if text_encoder_backend == "none":
            self.text_encoder = None
            obs_fusion_dim = hidden_size
        elif text_encoder_backend == "hash":
            self.text_encoder = HashTextEncoder(embedding_dim=text_embedding_dim)
            self.text_projection = nn.Sequential(
                nn.Linear(text_embedding_dim, hidden_size),
                nn.ReLU(),
            )
            obs_fusion_dim = hidden_size * 2
        else:
            raise ValueError(f"Unsupported text_encoder_backend: {text_encoder_backend}")

        self.obs_fusion = nn.Sequential(
            nn.Linear(obs_fusion_dim, hidden_size),
            nn.ReLU(),
        )
        self.init_posterior = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2),
        )
        self.action_embedding = nn.Embedding(action_dim, hidden_size)
        self.skill_embedding = nn.Embedding(skill_dim, hidden_size)
        self.gru = nn.GRUCell(hidden_size + latent_dim, hidden_size)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2),
        )
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_dim * 2),
        )
        predict_in = hidden_size + latent_dim
        self.feature_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.done_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.action_valid_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.planner_action_head = nn.Sequential(
            nn.Linear(predict_in, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def encode_texts(self, texts: list[str], *, device: torch.device) -> torch.Tensor | None:
        if self.text_encoder_backend == "none":
            return None
        raw = self.text_encoder([str(text or "") for text in texts], device=device)
        return self.text_projection(raw)

    def encode_observation(
        self,
        features: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_hidden = self.obs_encoder(features)
        if self.text_encoder_backend == "none":
            return self.obs_fusion(obs_hidden)
        if text_context is None:
            if prompt_texts is None:
                raise ValueError("prompt_texts or text_context is required when text conditioning is enabled")
            text_context = self.encode_texts(prompt_texts, device=features.device)
        return self.obs_fusion(torch.cat([obs_hidden, text_context], dim=-1))

    def observe(
        self,
        features: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs_embed = self.encode_observation(features, prompt_texts=prompt_texts, text_context=text_context)
        posterior_mean, posterior_logvar = _split_latent_stats(self.init_posterior(obs_embed))
        latent = _sample_gaussian(posterior_mean, posterior_logvar, deterministic=deterministic)
        hidden = torch.zeros(features.shape[0], self.hidden_size, device=features.device)
        return {
            "hidden": hidden,
            "latent": latent,
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "obs_embed": obs_embed,
        }

    def imagine_step(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor,
        tasks: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        transition_input = torch.cat(
            [self.action_embedding(actions) + self.skill_embedding(tasks), latent],
            dim=-1,
        )
        next_hidden = self.gru(transition_input, hidden)
        prior_mean, prior_logvar = _split_latent_stats(self.prior_net(next_hidden))
        prior_latent = _sample_gaussian(prior_mean, prior_logvar, deterministic=deterministic)
        belief = torch.cat([next_hidden, prior_latent], dim=-1)
        return {
            "hidden": next_hidden,
            "prior_mean": prior_mean,
            "prior_logvar": prior_logvar,
            "prior_latent": prior_latent,
            "pred_features": self.feature_head(belief),
            "pred_reward": self.reward_head(belief).squeeze(-1),
            "pred_done_logit": self.done_head(belief).squeeze(-1),
            "pred_action_valid_logits": self.action_valid_head(belief),
            "pred_value": self.value_head(belief).squeeze(-1),
        }

    def posterior_step(
        self,
        next_hidden: torch.Tensor,
        next_obs_embed: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        posterior_mean, posterior_logvar = _split_latent_stats(
            self.posterior_net(torch.cat([next_hidden, next_obs_embed], dim=-1))
        )
        posterior_latent = _sample_gaussian(posterior_mean, posterior_logvar, deterministic=deterministic)
        return {
            "posterior_mean": posterior_mean,
            "posterior_logvar": posterior_logvar,
            "posterior_latent": posterior_latent,
        }

    def forward_sequence(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        tasks: torch.Tensor,
        *,
        prompt_sequences: list[list[str]] | None = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        _, obs_len, _ = features.shape
        transition_len = actions.shape[1]
        if obs_len != transition_len + 1:
            raise ValueError(f"Expected obs_len = transition_len + 1, got obs_len={obs_len}, transitions={transition_len}")

        obs_embeds = []
        for idx in range(obs_len):
            prompt_texts = None if prompt_sequences is None else [seq[idx] for seq in prompt_sequences]
            obs_embeds.append(self.encode_observation(features[:, idx], prompt_texts=prompt_texts))

        initial = self.observe(features[:, 0], text_context=obs_embeds[0], deterministic=deterministic)
        hidden = initial["hidden"]
        latent = initial["latent"]

        prior_means = []
        prior_logvars = []
        posterior_means = [initial["posterior_mean"]]
        posterior_logvars = [initial["posterior_logvar"]]
        post_latents = [latent]
        pred_features = []
        pred_rewards = []
        pred_done_logits = []
        pred_action_valid_logits = []
        pred_values = []
        hidden_states = [hidden]

        for step in range(transition_len):
            imagined = self.imagine_step(hidden, latent, actions[:, step], tasks[:, step], deterministic=deterministic)
            prior_means.append(imagined["prior_mean"])
            prior_logvars.append(imagined["prior_logvar"])
            pred_features.append(imagined["pred_features"])
            pred_rewards.append(imagined["pred_reward"])
            pred_done_logits.append(imagined["pred_done_logit"])
            pred_action_valid_logits.append(imagined["pred_action_valid_logits"])
            pred_values.append(imagined["pred_value"])

            posterior = self.posterior_step(imagined["hidden"], obs_embeds[step + 1], deterministic=deterministic)
            hidden = imagined["hidden"]
            latent = posterior["posterior_latent"]
            hidden_states.append(hidden)
            posterior_means.append(posterior["posterior_mean"])
            posterior_logvars.append(posterior["posterior_logvar"])
            post_latents.append(latent)

        return {
            "prior_means": torch.stack(prior_means, dim=1),
            "prior_logvars": torch.stack(prior_logvars, dim=1),
            "posterior_means": torch.stack(posterior_means, dim=1),
            "posterior_logvars": torch.stack(posterior_logvars, dim=1),
            "posterior_latents": torch.stack(post_latents, dim=1),
            "hidden_states": torch.stack(hidden_states, dim=1),
            "pred_features": torch.stack(pred_features, dim=1),
            "pred_rewards": torch.stack(pred_rewards, dim=1),
            "pred_done_logits": torch.stack(pred_done_logits, dim=1),
            "pred_action_valid_logits": torch.stack(pred_action_valid_logits, dim=1),
            "pred_values": torch.stack(pred_values, dim=1),
            "planner_action_logits": self.planner_action_head(
                torch.cat([torch.stack(hidden_states[:-1], dim=1), torch.stack(post_latents[:-1], dim=1)], dim=-1)
            ),
        }


@dataclass
class SequenceWorldModelInference:
    model: SequenceWorldModel
    device: torch.device

    def _feature_tensor(self, features: np.ndarray | list[float] | list[list[float]]) -> torch.Tensor:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        return torch.from_numpy(feature_array).to(self.device)

    def observe(self, features: np.ndarray | list[list[float]], *, prompt_texts: list[str] | None = None) -> dict[str, np.ndarray]:
        feature_tensor = self._feature_tensor(features)
        with torch.no_grad():
            state = self.model.observe(feature_tensor, prompt_texts=prompt_texts, deterministic=True)
            belief = torch.cat([state["hidden"], state["latent"]], dim=-1)
            state["planner_action_logits"] = self.model.planner_action_head(belief)
        return {key: value.cpu().numpy() for key, value in state.items() if isinstance(value, torch.Tensor)}

    def rollout(
        self,
        initial_features: np.ndarray | list[float],
        actions: list[int],
        *,
        tasks: list[int] | None = None,
        prompt_text: str | None = None,
        deterministic: bool = True,
        num_samples: int = 1,
    ) -> dict[str, np.ndarray]:
        action_tensor = torch.tensor(actions, dtype=torch.long, device=self.device).reshape(1, -1)
        if tasks is None:
            tasks = [0] * len(actions)
        task_tensor = torch.tensor(tasks, dtype=torch.long, device=self.device).reshape(1, -1)
        feature_tensor = self._feature_tensor(initial_features)
        prompts = [prompt_text or ""]
        with torch.no_grad():
            sample_feature_rollouts = []
            sample_reward_rollouts = []
            sample_done_rollouts = []
            sample_value_rollouts = []
            sample_valid_rollouts = []
            sample_uncertainty_rollouts = []

            for _ in range(max(1, int(num_samples))):
                observed = self.model.observe(feature_tensor, prompt_texts=prompts, deterministic=deterministic)
                hidden = observed["hidden"]
                latent = observed["latent"]
                pred_features = []
                pred_rewards = []
                pred_done_logits = []
                pred_values = []
                pred_action_valid_logits = []
                pred_latent_uncertainties = []
                for step in range(action_tensor.shape[1]):
                    imagined = self.model.imagine_step(
                        hidden,
                        latent,
                        action_tensor[:, step],
                        task_tensor[:, step],
                        deterministic=deterministic,
                    )
                    hidden = imagined["hidden"]
                    latent = imagined["prior_latent"]
                    pred_features.append(imagined["pred_features"])
                    pred_rewards.append(imagined["pred_reward"])
                    pred_done_logits.append(imagined["pred_done_logit"])
                    pred_values.append(imagined["pred_value"])
                    pred_action_valid_logits.append(imagined["pred_action_valid_logits"])
                    pred_latent_uncertainties.append(torch.exp(imagined["prior_logvar"]).mean(dim=-1))
                sample_feature_rollouts.append(torch.cat(pred_features, dim=0))
                sample_reward_rollouts.append(torch.cat(pred_rewards, dim=0))
                sample_done_rollouts.append(torch.sigmoid(torch.cat(pred_done_logits, dim=0)))
                sample_value_rollouts.append(torch.cat(pred_values, dim=0))
                sample_valid_rollouts.append(torch.sigmoid(torch.cat(pred_action_valid_logits, dim=0)))
                sample_uncertainty_rollouts.append(torch.cat(pred_latent_uncertainties, dim=0))

            feature_samples = torch.stack(sample_feature_rollouts, dim=0)
            reward_samples = torch.stack(sample_reward_rollouts, dim=0)
            done_samples = torch.stack(sample_done_rollouts, dim=0)
            value_samples = torch.stack(sample_value_rollouts, dim=0)
            valid_samples = torch.stack(sample_valid_rollouts, dim=0)
            uncertainty_samples = torch.stack(sample_uncertainty_rollouts, dim=0)
            reward_disagreement = reward_samples.std(dim=0) if feature_samples.shape[0] > 1 else torch.zeros_like(reward_samples[0])
            value_disagreement = value_samples.std(dim=0) if feature_samples.shape[0] > 1 else torch.zeros_like(value_samples[0])
        return {
            "pred_features": feature_samples.mean(dim=0).cpu().numpy(),
            "pred_rewards": reward_samples.mean(dim=0).cpu().numpy(),
            "pred_done_probs": done_samples.mean(dim=0).cpu().numpy(),
            "pred_values": value_samples.mean(dim=0).cpu().numpy(),
            "pred_action_valid_probs": valid_samples.mean(dim=0).cpu().numpy(),
            "pred_latent_uncertainty": uncertainty_samples.mean(dim=0).cpu().numpy(),
            "reward_disagreement": reward_disagreement.cpu().numpy(),
            "value_disagreement": value_disagreement.cpu().numpy(),
            "num_samples": int(max(1, int(num_samples))),
            "deterministic": bool(deterministic),
        }


def save_sequence_world_model(model: SequenceWorldModel, path: str, metadata: dict | None = None) -> None:
    atomic_torch_save(path, {"state_dict": model.state_dict(), "metadata": metadata or {}})


def load_sequence_world_model(path: str, device: str = "cpu") -> SequenceWorldModelInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    model = SequenceWorldModel(
        input_dim=int(metadata["input_dim"]),
        latent_dim=int(metadata.get("latent_dim", 64)),
        hidden_size=int(metadata.get("hidden_size", 128)),
        text_encoder_backend=metadata.get("text_encoder_backend", "none"),
        text_embedding_dim=int(metadata.get("text_embedding_dim", 64)),
    )
    model.load_state_dict(payload["state_dict"], strict=False)
    model._metadata = metadata
    model.to(torch_device)
    model.eval()
    return SequenceWorldModelInference(model=model, device=torch_device)
