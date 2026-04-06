from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import torch
from torch import nn

from rl.feature_encoder import ACTION_SET
from rl.io_utils import atomic_torch_save


class StableHashTextEncoder(nn.Module):
    def __init__(self, vocab_size: int = 4096, embedding_dim: int = 128):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.EmbeddingBag(self.vocab_size, self.embedding_dim, mode="mean")

    def _hash_text(self, text: str) -> list[int]:
        tokens = [token for token in str(text).lower().split() if token]
        if not tokens:
            return [0]
        token_ids: list[int] = []
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            token_ids.append(int.from_bytes(digest, "little") % self.vocab_size)
        return token_ids

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


class BCPolicyMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        output_dim: int = len(ACTION_SET),
        num_layers: int = 2,
        text_encoder_backend: str = "none",
        text_vocab_size: int = 4096,
        text_embedding_dim: int = 128,
        text_model_name: str | None = None,
        text_max_length: int = 128,
        text_trainable: bool = False,
    ):
        super().__init__()
        self.text_encoder_backend = str(text_encoder_backend or "none")
        self.text_model_name = text_model_name
        self.text_max_length = int(text_max_length)
        self.text_trainable = bool(text_trainable)
        resolved_layers = max(int(num_layers), 1)
        if self.text_encoder_backend == "none":
            layers: list[nn.Module] = []
            prev_dim = input_dim
            for _ in range(resolved_layers):
                layers.append(nn.Linear(prev_dim, hidden_size))
                layers.append(nn.ReLU())
                prev_dim = hidden_size
            layers.append(nn.Linear(prev_dim, output_dim))
            self.net = nn.Sequential(*layers)
            self.feature_stem = None
            self.text_encoder = None
            self.text_projection = None
            self.fusion = None
            self.hidden_tower = None
            self.output_head = None
        elif self.text_encoder_backend == "hash":
            self.net = None
            self.feature_stem = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            self.text_encoder = StableHashTextEncoder(
                vocab_size=text_vocab_size,
                embedding_dim=text_embedding_dim,
            )
            self.text_tokenizer = None
            self.text_projection = nn.Sequential(
                nn.Linear(text_embedding_dim, hidden_size),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
            )
            tower_layers: list[nn.Module] = []
            for _ in range(max(resolved_layers - 1, 0)):
                tower_layers.append(nn.Linear(hidden_size, hidden_size))
                tower_layers.append(nn.ReLU())
            self.hidden_tower = nn.Sequential(*tower_layers) if tower_layers else None
            self.output_head = nn.Linear(hidden_size, output_dim)
        elif self.text_encoder_backend == "transformer":
            from transformers import AutoModel, AutoTokenizer

            if not text_model_name:
                raise ValueError("text_model_name is required when text_encoder_backend='transformer'")
            self.net = None
            self.feature_stem = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
            )
            self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            self.text_encoder = AutoModel.from_pretrained(text_model_name)
            text_hidden_dim = int(getattr(self.text_encoder.config, "hidden_size"))
            if not self.text_trainable:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            self.text_projection = nn.Sequential(
                nn.Linear(text_hidden_dim, hidden_size),
                nn.ReLU(),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
            )
            tower_layers = []
            for _ in range(max(resolved_layers - 1, 0)):
                tower_layers.append(nn.Linear(hidden_size, hidden_size))
                tower_layers.append(nn.ReLU())
            self.hidden_tower = nn.Sequential(*tower_layers) if tower_layers else None
            self.output_head = nn.Linear(hidden_size, output_dim)
        else:
            raise ValueError(f"Unsupported BC text_encoder_backend: {self.text_encoder_backend}")

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
        texts = [str(text or "") for text in prompt_texts]
        if self.text_encoder_backend == "hash":
            raw = self.text_encoder(texts, device=device)
        elif self.text_encoder_backend == "transformer":
            raw = self._encode_transformer_text(texts, device=device)
        else:
            raise ValueError("encode_text_context called without a text encoder")
        return self.text_projection(raw)

    def forward(
        self,
        x: torch.Tensor,
        *,
        prompt_texts: list[str] | None = None,
        text_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.text_encoder_backend == "none":
            return self.net(x)
        if text_context is None:
            normalized_prompts = [str(text or "") for text in (prompt_texts or [""] * x.shape[0])]
            if len(normalized_prompts) != int(x.shape[0]):
                raise ValueError(f"Expected {x.shape[0]} prompt_texts, got {len(normalized_prompts)}")
            text_context = self.encode_text_context(normalized_prompts, device=x.device)
        feature_hidden = self.feature_stem(x)
        hidden = self.fusion(torch.cat([feature_hidden, text_context], dim=-1))
        if self.hidden_tower is not None:
            hidden = self.hidden_tower(hidden)
        return self.output_head(hidden)


@dataclass
class BCPolicyInference:
    model: BCPolicyMLP
    device: torch.device
    metadata: dict

    def _masked_logits(
        self,
        feature_tensor: torch.Tensor,
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
        prompt_texts: list[str] | None = None,
    ) -> torch.Tensor:
        logits = self.model(feature_tensor, prompt_texts=prompt_texts)
        if allowed_actions_list:
            for row_idx, allowed_actions in enumerate(allowed_actions_list):
                if not allowed_actions:
                    continue
                allowed = set(allowed_actions)
                for idx, name in enumerate(ACTION_SET):
                    if name not in allowed:
                        logits[row_idx, idx] = -1e9
        return logits

    def act(
        self,
        features: np.ndarray,
        allowed_actions: list[str] | None = None,
        *,
        prompt_text: str | None = None,
    ) -> str:
        action_idx = self.act_batch(
            [features],
            allowed_actions_list=[allowed_actions],
            prompt_texts=[prompt_text or ""],
        )[0]
        return ACTION_SET[action_idx]

    def logits_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
        prompt_texts: list[str] | None = None,
    ) -> np.ndarray:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        normalized_prompts = None
        if prompt_texts is not None:
            if len(prompt_texts) != feature_array.shape[0]:
                raise ValueError(f"Expected {feature_array.shape[0]} prompt_texts, got {len(prompt_texts)}")
            normalized_prompts = [text or "" for text in prompt_texts]
        with torch.no_grad():
            logits = self._masked_logits(
                torch.from_numpy(feature_array).to(self.device),
                allowed_actions_list=allowed_actions_list,
                prompt_texts=normalized_prompts,
            )
        return logits.cpu().numpy()

    def act_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
        prompt_texts: list[str] | None = None,
    ) -> list[int]:
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim == 1:
            feature_array = feature_array.reshape(1, -1)
        normalized_prompts = None
        if prompt_texts is not None:
            if len(prompt_texts) != feature_array.shape[0]:
                raise ValueError(f"Expected {feature_array.shape[0]} prompt_texts, got {len(prompt_texts)}")
            normalized_prompts = [text or "" for text in prompt_texts]
        with torch.no_grad():
            logits = self._masked_logits(
                torch.from_numpy(feature_array).to(self.device),
                allowed_actions_list=allowed_actions_list,
                prompt_texts=normalized_prompts,
            )
            action_indices = torch.argmax(logits, dim=1).tolist()
        return [int(idx) for idx in action_indices]

    def act_names_batch(
        self,
        features: np.ndarray | list[list[float]],
        *,
        allowed_actions_list: list[list[str] | None] | None = None,
        prompt_texts: list[str] | None = None,
    ) -> list[str]:
        return [
            ACTION_SET[idx]
            for idx in self.act_batch(
                features,
                allowed_actions_list=allowed_actions_list,
                prompt_texts=prompt_texts,
            )
        ]


def save_bc_model(model: BCPolicyMLP, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    atomic_torch_save(path, payload)


def load_bc_model(path: str, input_dim: int | None = None, device: str = "cpu") -> BCPolicyInference:
    torch_device = torch.device(device)
    payload = torch.load(path, map_location=torch_device)
    metadata = payload.get("metadata", {})
    resolved_input_dim = input_dim or metadata.get("input_dim")
    if resolved_input_dim is None:
        raise ValueError("BC model input_dim is required and was not found in metadata")
    hidden_size = metadata.get("hidden_size", 256)
    num_layers = metadata.get("num_layers", 2)
    model = BCPolicyMLP(
        input_dim=resolved_input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        text_encoder_backend=metadata.get("text_encoder_backend", "none"),
        text_vocab_size=metadata.get("text_vocab_size", 4096),
        text_embedding_dim=metadata.get("text_embedding_dim", 128),
        text_model_name=metadata.get("text_model_name"),
        text_max_length=metadata.get("text_max_length", 128),
        text_trainable=metadata.get("text_trainable", False),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.to(torch_device)
    return BCPolicyInference(model=model, device=torch_device, metadata=metadata)
