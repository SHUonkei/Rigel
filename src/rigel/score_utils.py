"""Score utilities: convert class logits to a continuous score (score smoothing)."""

import torch


def logits_to_score(logits: torch.Tensor, num_classes: int = 5) -> torch.Tensor:
    """Convert logits to a continuous score via the softmax expected value.

    Class i is mapped to the value i / (num_classes - 1), so for num_classes=5 the
    class values are [0.0, 0.25, 0.5, 0.75, 1.0].

    Args:
        logits: (batch_size, num_classes) classification logits.
        num_classes: Number of score classes.

    Returns:
        (batch_size,) scores in [0, 1].
    """
    probs = torch.softmax(logits, dim=-1)
    class_values = torch.arange(num_classes, dtype=torch.float32, device=logits.device)
    class_values = class_values / (num_classes - 1)
    return (probs * class_values).sum(dim=-1)
