"""Composite Dataset Evaluation Script

Usage:
    # (with references)
    uv run python src/evaluate/composite.py \
        --data_root data/composite \
        --output_dir results/composite

    # (no references)
    uv run python src/evaluate/composite.py \
        --data_root data/composite \
        --output_dir results/composite_noref \
        --no_references
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from scipy.stats import kendalltau
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src/ to path so the scripts also run without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rigel import QwenLoRAScoreSmoothing, load_checkpoint_strict, resolve_checkpoint


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    """Load Rigel model for image evaluation."""
    cfg = OmegaConf.load(str(config_path))

    model = QwenLoRAScoreSmoothing(cfg)

    if checkpoint_path and checkpoint_path.exists():
        load_checkpoint_strict(model, checkpoint_path, map_location=device, verbose=True)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model.eval()
    model.to(device)
    return model


def compute_correlations(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """Compute correlation coefficients (Kendall tau_b, tau_c, multiplied by 100)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        kendall_tau_b, kendall_p_b = kendalltau(predictions, ground_truth, variant="b")
        kendall_tau_c, kendall_p_c = kendalltau(predictions, ground_truth, variant="c")

        return {
            "kendall_tau_b": float(kendall_tau_b * 100),
            "kendall_tau_b_p_value": float(kendall_p_b),
            "kendall_tau_c": float(kendall_tau_c * 100),
            "kendall_tau_c_p_value": float(kendall_p_c),
        }


class CompositeDataset:
    """Composite dataset loader (Flickr8k + COCO images)."""

    def __init__(self, data_root: Path, use_references: bool = True):
        self.data_root = Path(data_root)
        self.use_references = use_references

        self.csv_path = self.data_root / "en_test_composite_da2.csv"
        self.images_dir = self.data_root / "images"

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        self.df = pd.read_csv(self.csv_path)
        print(f"✓ Composite Dataset: {len(self.df)} samples")

    def __len__(self):
        return len(self.df)

    def get_batch(self, idx: int, batch_size: int):
        """Get batch data."""
        batch_df = self.df.iloc[idx:idx + batch_size]

        image_ids = batch_df["imgid"].tolist()
        cands = batch_df["mt"].astype(str).tolist()
        refs_raw = batch_df["refs"].tolist()
        scores = batch_df["score"].tolist()

        if self.use_references:
            refs = []
            for ref_str in refs_raw:
                try:
                    ref_list = eval(ref_str)
                    refs.append(ref_list if isinstance(ref_list, list) else [])
                except Exception:
                    refs.append([])
        else:
            refs = [[] for _ in range(len(refs_raw))]

        image_paths = [str(self.images_dir / img_id) for img_id in image_ids]
        valid_indices = [i for i, p in enumerate(image_paths) if os.path.exists(p)]

        if not valid_indices:
            return None

        return {
            'image_ids': [image_ids[i] for i in valid_indices],
            'image_paths': [image_paths[i] for i in valid_indices],
            'candidates': [cands[i] for i in valid_indices],
            'references': [refs[i] for i in valid_indices],
            'scores': [scores[i] for i in valid_indices]
        }


def evaluate_composite(model, dataset: CompositeDataset, device: torch.device,
                      batch_size: int = 8) -> Dict:
    """Evaluate on Composite dataset."""
    print("\n" + "=" * 80)
    ref_mode = "with references" if dataset.use_references else "without references"
    print(f"Composite Dataset Evaluation ({ref_mode})")
    print("=" * 80)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for idx in tqdm(range(0, len(dataset), batch_size), desc="Composite evaluation"):
            batch = dataset.get_batch(idx, batch_size)
            if batch is None:
                continue

            try:
                pred_scores = model(
                    media_id=batch['image_ids'],
                    media_path=batch['image_paths'],
                    cand=batch['candidates'],
                    refs=batch['references'],
                    modality=["image"] * len(batch['image_ids'])
                )

                if torch.is_tensor(pred_scores):
                    pred_scores = pred_scores.cpu().numpy()

                predictions.extend(pred_scores)
                ground_truth.extend(batch['scores'])

            except Exception as e:
                print(f"  ⚠ Inference error: {e}")
                continue

    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    correlations = compute_correlations(predictions, ground_truth)

    print("\nResults:")
    print(f"  Kendall Tau-b: {correlations['kendall_tau_b']:.2f}")
    print(f"  Kendall Tau-c: {correlations['kendall_tau_c']:.2f} (primary metric)")

    return {
        "num_samples": len(predictions),
        "correlations": correlations,
        "predictions_mean": float(predictions.mean()),
        "predictions_std": float(predictions.std()),
        "ground_truth_mean": float(ground_truth.mean()),
        "ground_truth_std": float(ground_truth.std()),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Composite Dataset Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Local weights (.safetensors/.pth). Defaults to the Hugging Face release")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Local config YAML. Defaults to the Hugging Face release")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing Composite data")
    parser.add_argument("--output_dir", type=str, default="results/composite",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--no_references", action="store_true",
                       help="Evaluate without references")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    checkpoint_path, config_path = resolve_checkpoint(args.checkpoint_path, args.config_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, config_path, device)
    dataset = CompositeDataset(data_root, use_references=not args.no_references)
    results = evaluate_composite(model, dataset, device, args.batch_size)

    output_file = output_dir / f"composite_{'ref' if not args.no_references else 'noref'}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "use_references": not args.no_references,
                "batch_size": args.batch_size
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
