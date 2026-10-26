"""Checkpoint loading utilities."""

from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file


def load_state_dict(
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    """Read a ``.safetensors`` or ``.pth`` checkpoint into a flat state dict.

    Args:
        checkpoint_path: Path to the checkpoint. A ``.pth`` file may hold either a raw
            state dict or a dict with a ``model_state_dict`` entry.
        map_location: Device to load tensors onto.

    Returns:
        The state dict with any ``module.`` (DataParallel) prefix removed.

    Raises:
        FileNotFoundError: The checkpoint file does not exist.
        RuntimeError: The checkpoint is empty or not a state dict.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path), device=str(map_location))
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location=map_location, weights_only=False)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        raise RuntimeError(f"Invalid or empty checkpoint: {checkpoint_path}")

    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def load_checkpoint_strict(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    map_location: Union[str, torch.device] = "cpu",
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load a LoRA + head checkpoint into ``model``.

    Every key in the checkpoint must exist in the model. Keys that exist in the
    model but not in the checkpoint (the frozen Qwen3-VL base weights) are allowed.

    Args:
        model: Model to load the weights into.
        checkpoint_path: Path to the ``.safetensors`` or ``.pth`` file.
        map_location: Device to load tensors onto.
        verbose: Print a summary of the loaded keys.

    Returns:
        The state dict that was loaded.

    Raises:
        FileNotFoundError: The checkpoint file does not exist.
        RuntimeError: The checkpoint contains keys the model does not have, or no
            key could be loaded.
    """
    state_dict = load_state_dict(checkpoint_path, map_location=map_location)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint contains keys that do not exist in the model "
            f"({len(incompatible.unexpected_keys)} keys). First ones: "
            f"{incompatible.unexpected_keys[:10]}"
        )

    loaded_keys = set(state_dict) - set(incompatible.missing_keys)
    if not loaded_keys:
        raise RuntimeError("No checkpoint keys matched the model.")

    if verbose:
        lora_keys = [k for k in loaded_keys if "lora" in k.lower()]
        head_keys = [k for k in loaded_keys if k.startswith(("encoder.", "decoder.", "head."))]
        print(f"Checkpoint loaded from: {checkpoint_path}")
        print(f"  loaded {len(loaded_keys)} / {len(state_dict)} keys "
              f"(LoRA: {len(lora_keys)}, head: {len(head_keys)})")

    return state_dict
