"""Rigel: caption quality scoring with Qwen3-VL + LoRA + bottleneck head."""

from .checkpoint import load_checkpoint_strict, load_state_dict
from .hub import DEFAULT_REPO_ID, from_pretrained, resolve_checkpoint
from .prompt_templates import create_prompt
from .qwen_lora_score_smoothing import QwenLoRAScoreSmoothing
from .score_utils import logits_to_score

__all__ = [
    "DEFAULT_REPO_ID",
    "QwenLoRAScoreSmoothing",
    "create_prompt",
    "from_pretrained",
    "load_checkpoint_strict",
    "load_state_dict",
    "logits_to_score",
    "resolve_checkpoint",
]
