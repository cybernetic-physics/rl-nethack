from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SkillConditionedModelSpec:
    """Model spec only. Actual implementation should remain swappable."""

    backbone: str
    hidden_size: int
    num_layers: int
    use_lstm: bool
    lstm_size: int
    skill_embedding_dim: int
    share_backbone_across_skills: bool


def build_model_spec(config) -> SkillConditionedModelSpec:
    return SkillConditionedModelSpec(
        backbone=config.backbone,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        use_lstm=config.use_lstm,
        lstm_size=config.lstm_size,
        skill_embedding_dim=config.skill_embedding_dim,
        share_backbone_across_skills=config.share_backbone_across_skills,
    )
