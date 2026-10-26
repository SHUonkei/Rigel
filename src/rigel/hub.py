"""Download the released checkpoint from the Hugging Face Hub and build the model."""

from pathlib import Path
from typing import Optional, Tuple, Union

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

from .checkpoint import load_checkpoint_strict
from .qwen_lora_score_smoothing import QwenLoRAScoreSmoothing

DEFAULT_REPO_ID = "ShuitsuKoyama/Rigel"
WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.yaml"


def resolve_checkpoint(
    checkpoint_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Return local paths to the weights and config, downloading what is not given.

    Args:
        checkpoint_path: Local ``.safetensors`` / ``.pth`` file. Downloaded from the Hub if None.
        config_path: Local config YAML. Downloaded from the Hub if None.
        repo_id: Hugging Face model repository to download from.
        revision: Branch, tag or commit of the repository.

    Returns:
        (weights path, config path).
    """
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(repo_id, WEIGHTS_NAME, revision=revision)
    if config_path is None:
        config_path = hf_hub_download(repo_id, CONFIG_NAME, revision=revision)
    # absolute() rather than resolve(): Hub cache files are symlinks to extension-less blobs,
    # and the weights format is detected from the suffix.
    return Path(checkpoint_path).expanduser().absolute(), Path(config_path).expanduser().absolute()


def from_pretrained(
    repo_id: str = DEFAULT_REPO_ID,
    revision: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> QwenLoRAScoreSmoothing:
    """Build the scorer and load the released (or a local) checkpoint.

    Args:
        repo_id: Hugging Face model repository to download from.
        revision: Branch, tag or commit of the repository.
        checkpoint_path: Local weights; overrides the Hub download.
        config_path: Local config YAML; overrides the Hub download.
        verbose: Print a summary of the loaded keys.

    Returns:
        The model in eval mode on the GPU.
    """
    weights, config = resolve_checkpoint(checkpoint_path, config_path, repo_id, revision)
    cfg = OmegaConf.load(str(config))
    model = QwenLoRAScoreSmoothing(cfg)
    load_checkpoint_strict(model, weights, map_location=model.device, verbose=verbose)
    model.eval()
    return model
