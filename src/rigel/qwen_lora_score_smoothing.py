"""QwenLoRAScoreSmoothing: Qwen3-VL + LoRA + bottleneck head (inference only).

Architecture:
    Qwen3-VL (LoRA on q_proj / v_proj)
        -> last hidden state of the final token
        -> bottleneck head (encoder + decoder) -> num_classes logits
        -> softmax -> expected value in [0, 1]

All trainable weights (LoRA adapters and the head) are stored in the checkpoint
and loaded with ``utils.load_checkpoint_strict``.
"""

import os
from typing import List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .prompt_templates import create_prompt
from .score_utils import logits_to_score

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def infer_modality(media_path: str) -> str:
    """Guess "image" or "video" from the file extension."""
    return "image" if os.path.splitext(media_path)[1].lower() in IMAGE_SUFFIXES else "video"


class BottleneckHead(nn.Module):
    """Encoder (hidden -> bottleneck) followed by a decoder (bottleneck -> classes)."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class QwenLoRAScoreSmoothing(nn.Module):
    """Caption quality scorer built on Qwen3-VL.

    Args:
        cfg: Configuration with the keys documented in ``configs/inference.yaml``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name: str = cfg.get("model_name", "Qwen/Qwen3-VL-2B-Instruct")
        print(f"Loading Qwen3-VL model: {model_name}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
        )

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "left"

        self.num_frames: int = cfg.get("num_frames", 8)
        if hasattr(self.processor, "video_processor"):
            self.processor.video_processor.num_frames = self.num_frames
            self.processor.video_processor.fps = None

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.get("lora_r", 8),
            lora_alpha=cfg.get("lora_alpha", 16),
            lora_dropout=cfg.get("lora_dropout", 0.0),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
        )
        self.model = get_peft_model(self.model, lora_config)

        self.hidden_size: int = self.model.config.text_config.hidden_size
        self.num_classes: int = cfg.get("num_classes", 5)
        self._build_bottleneck_head(bottleneck_dim=cfg.get("bottleneck_dim", 256))

        for param in self.parameters():
            param.requires_grad = False

    def _build_bottleneck_head(self, bottleneck_dim: int) -> None:
        """Build the head. Weights are supplied by the checkpoint.

        ``self.encoder`` / ``self.decoder`` are registered directly on this module and
        also wrapped in ``self.head``; the checkpoint stores both key prefixes.
        """
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim * 2),
            nn.LayerNorm(bottleneck_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_dim * 2, self.num_classes),
        )
        self.encoder.to(self.device).to(torch.bfloat16)
        self.decoder.to(self.device).to(torch.bfloat16)
        self.head = BottleneckHead(self.encoder, self.decoder)

    def _build_messages(
        self,
        media_path: List[str],
        cand: List[str],
        refs: List[List[str]],
        modality: List[str],
    ) -> List[List[dict]]:
        all_messages = []
        for path, candidate, references, current_modality in zip(media_path, cand, refs, modality):
            prompt = create_prompt(candidate, references, modality=current_modality)
            abs_media_path = os.path.abspath(path)
            content = [
                {"type": current_modality, current_modality: abs_media_path},
                {"type": "text", "text": prompt},
            ]
            all_messages.append([{"role": "user", "content": content}])
        return all_messages

    @torch.no_grad()
    def forward(
        self,
        media_path: List[str],
        cand: List[str],
        refs: Optional[List[List[str]]] = None,
        modality: Optional[List[str]] = None,
        media_id: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Score candidate captions.

        Args:
            media_path: Video or image file path per sample.
            cand: Candidate caption per sample.
            refs: Reference captions per sample (empty list for reference-free scoring).
            modality: "video" or "image" per sample. Guessed from the file extension if None.
            media_id: Unused; accepted so callers can pass sample identifiers.

        Returns:
            Tensor of shape (batch_size,) with continuous scores in [0, 1].
        """
        batch_size = len(media_path)
        if refs is None:
            refs = [[] for _ in range(batch_size)]
        if modality is None:
            modality = [infer_modality(path) for path in media_path]

        all_messages = self._build_messages(media_path, cand, refs, modality)

        template_kwargs = dict(
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        max_pixels = self.cfg.get("max_pixels", None)
        if max_pixels is not None:
            template_kwargs["max_pixels"] = max_pixels
        inputs = self.processor.apply_chat_template(all_messages, **template_kwargs)
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        model_outputs = self.model.base_model.model.model(**inputs)
        last_hidden_state = model_outputs.last_hidden_state[:, -1, :]
        logits = self.head(last_hidden_state)
        return logits_to_score(logits, num_classes=self.num_classes)
