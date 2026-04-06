from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET, SKILL_SET
from rl.io_utils import atomic_torch_save


class HashTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")

    def _hash_text(self, text: str) -> list[int]:
        tokens = [token for token in text.lower().split() if token]
        if not tokens:
            return [0]
        return [abs(hash(token)) % self.vocab_size for token in tokens]

    def forward(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        indices: list[int] = []
        offsets: list[int] = []
        offset = 0
        for text in texts:
            token_ids = self._hash_text(text)
            offsets.append(offset)
            indices.extend(token_ids)
            offset += len(token_ids)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        return self.embedding(index_tensor, offset_tensor)


class TraceWorldModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        hidden_size: int = 256,
        action_dim: int = len(ACTION_SET),
        skill_dim: int = len(SKILL_SET),
        text_encoder_backend: str = "none",
        text_model_name: str | None = None,
        text_max_length: int = 128,
        text_trainable: bool = False,
        text_embedding_dim: int = 128,
    ):
        super().__init__()
        self.text_encoder_backend = text_encoder_backend
        self.text_model_name = text_model_name
        self.text_max_length = text_max_length
        self.text_trainable = text_trainable
        self.text_embedding_dim = text_embedding_dim

        if text_encoder_backend == "none":
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_dim),
                nn.ReLU(),
            )
            self.feature_stem = None
            self.text_encoder = None
            self.text_tokenizer = None
            self.text_projection = None
            self.fusion = None
        else:
            self.encoder = None
            self.feature_stem = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            if text_encoder_backend == "hash":
                self.text_encoder = HashTextEncoder(embedding_dim=text_embedding_dim)
                text_hidden_dim = text_embedding_dim
                self.text_tokenizer = None
            elif text_encoder_backend == "transformer":
                from transformers import AutoModel, AutoTokenizer

                if not text_model_name:
                    raise ValueError("text_model_name is required when text_encoder_backend='transformer'")
                self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
                self.text_encoder = AutoModel.from_pretrained(text_model_name)
                text_hidden_dim = int(getattr(self.text_encoder.config, "hidden_size"))
                if not text_trainable:
                    for param in self.text_encoder.parameters():
                        param.requires_grad = False
            else:
                raise ValueError(f"Unsupported text_encoder_backend: {text_encoder_backend}")
            self.text_projection = nn.Sequential(
                nn.Linear(text_hidden_dim, hidden_size),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, latent_dim),
                nn.ReLU(),
            )
        self.action_embedding = nn.Embedding(action_dim, latent_dim)
        self.skill_embedding = nn.Embedding(skill_dim, latent_dim)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.current_feature_head = nn.Linear(latent_dim, input_dim)
        self.action_head = nn.Linear(latent_dim, action_dim)
        self.future_feature_head = nn.Linear(hidden_size, input_dim)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.done_head = nn.Linear(hidden_size, 1)

    def _encode_transformer_text(self, texts: list[str], *, device: torch.device) -> torch.Tensor:
        encoded = self.text_tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.text_max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.text_encoder(**encoded)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            hidden = outputs.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def encode_text_context(self, prompt_texts: list[str], *, device: torch.device) -> torch.Tensor:
        texts = [str(text) for text in prompt_texts]
        if self.text_encoder_backend == "none":
            raise ValueError("encode_text_context called without a text encoder")
        if self.text_encoder_backend == "hash":
            raw = self.text_encoder(texts, device=device)
        elif self.text_encoder_backend == "transformer":
            raw = self._encode_transformer_text(texts, device=device)
        else:
            raise ValueError(f"Unsupported text_encoder_backend: {self.text_encoder_backend}")
        return self.text_projection(raw)

    def encode(
        self,
        features: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.text_encoder_backend == "none":
            return self.encoder(features)
        if text_context is None:
            if prompt_texts is None:
                raise ValueError("prompt_texts or text_context is required for text-conditioned world models")
            text_context = self.encode_text_context(prompt_texts, device=features.device)
        feature_hidden = self.feature_stem(features)
        return self.fusion(torch.cat([feature_hidden, text_context], dim=-1))

    def forward(
        self,
        features: torch.Tensor,
        actions: torch.Tensor,
        tasks: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(features, prompt_texts=prompt_texts, text_context=text_context)
        action_latent = self.action_embedding(actions)
        task_latent = self.skill_embedding(tasks)
        hidden = self.transition(torch.cat([latent, action_latent, task_latent], dim=-1))
        return {
            "latent": latent,
            "hidden": hidden,
            "current_features": self.current_feature_head(latent),
            "action_logits": self.action_head(latent),
            "future_features": self.future_feature_head(hidden),
            "reward": self.reward_head(hidden).squeeze(-1),
            "done_logit": self.done_head(hidden).squeeze(-1),
        }


@dataclass
class WorldModelInference:
    model: TraceWorldModel
    device: torch.device

    def encode(self, features: np.ndarray, *, prompt_text: str | None = None) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        with torch.no_grad():
            latent = self._encode_tensor(
                torch.from_numpy(features).unsqueeze(0).to(self.device),
                prompt_texts=[prompt_text or ""],
            )
        return latent.squeeze(0).cpu().numpy()

    def _encode_tensor(
        self,
        feature_tensor: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = self.model.encode(feature_tensor, prompt_texts=prompt_texts, text_context=text_context)
        metadata = getattr(self.model, "_metadata", {})
        mean = metadata.get("latent_mean")
        std = metadata.get("latent_std")
        if mean is not None and std is not None:
            mean_t = torch.tensor(mean, dtype=torch.float32, device=self.device)
            std_t = torch.tensor(std, dtype=torch.float32, device=self.device).clamp_min(1e-6)
            latent = (latent - mean_t.unsqueeze(0)) / std_t.unsqueeze(0)
        return latent

    def encode_with_aux(self, features: np.ndarray, *, prompt_text: str | None = None) -> dict[str, np.ndarray]:
        features = np.asarray(features, dtype=np.float32)
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self._encode_tensor(feature_tensor, prompt_texts=[prompt_text or ""])
            action_logits = self.model.action_head(latent)
            current_features = self.model.current_feature_head(latent)
        return {
            "latent": latent.squeeze(0).cpu().numpy(),
            "action_logits": action_logits.squeeze(0).cpu().numpy(),
            "current_features": current_features.squeeze(0).cpu().numpy(),
        }


def save_world_model(model: TraceWorldModel, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    atomic_torch_save(path, payload)


def load_world_model(path: str, device: str = "cpu") -> WorldModelInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    input_dim = metadata.get("input_dim")
    if input_dim is None:
        raise ValueError("World model input_dim missing from metadata")
    model = TraceWorldModel(
        input_dim=input_dim,
        latent_dim=metadata.get("latent_dim", 128),
        hidden_size=metadata.get("hidden_size", 256),
        text_encoder_backend=metadata.get("text_encoder_backend", "none"),
        text_model_name=metadata.get("text_model_name"),
        text_max_length=metadata.get("text_max_length", 128),
        text_trainable=metadata.get("text_trainable", False),
        text_embedding_dim=metadata.get("text_embedding_dim", 128),
    )
    model.load_state_dict(payload["state_dict"])
    model._metadata = metadata
    model.to(torch_device)
    model.eval()
    return WorldModelInference(model=model, device=torch_device)
