"""Flickr8k-CF & Expert Dataset Evaluation Script

Usage:
    # (expert variant)
    uv run python src/evaluate/flickr8k.py \
        --data_root data/flickr8k \
        --output_dir results/flickr8k \
        --variant expert

    # (cf variant, no references)
    uv run python src/evaluate/flickr8k.py \
        --data_root data/flickr8k \
        --output_dir results/flickr8k \
        --variant cf \
        --no_references

    # All variants at once
    uv run python src/evaluate/flickr8k.py \
        --data_root data/flickr8k \
        --output_dir results/flickr8k \
        --all
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
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


def compute_correlations(predictions: np.ndarray, ground_truth: np.ndarray,
                        kendall_variant: str = "c") -> Dict[str, float]:
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
            "primary_metric": kendall_variant
        }


class Flickr8kDataset:
    """Flickr8k dataset loader (CF or Expert variant)."""

    def __init__(self, data_root: Path, variant: str = "expert", use_references: bool = True):
        self.data_root = Path(data_root)
        self.variant = variant
        self.use_references = use_references

        if variant == "expert":
            self.json_file = self.data_root / "flickr8k.json"
            self.kendall_variant = "c"
        elif variant == "cf":
            self.json_file = self.data_root / "crowdflower_flickr8k.json"
            self.kendall_variant = "b"
        else:
            raise ValueError(f"Invalid variant: {variant}. Must be 'expert' or 'cf'")

        self.img_dir = self.data_root / "images"

        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_file}")
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        with open(self.json_file, 'r') as f:
            data = json.load(f)

        self.samples = []
        for vid, vdata in data.items():
            image_filename = os.path.basename(vdata['image_path'])
            ground_truth_captions = vdata['ground_truth']

            for human_judgment in vdata['human_judgement']:
                if np.isnan(human_judgment['rating']):
                    continue

                self.samples.append({
                    'image_filename': image_filename,
                    'candidate': human_judgment['caption'],
                    'references': ground_truth_captions,
                    'score': human_judgment['rating']
                })

        print(f"✓ Flickr8k {variant.upper()} Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def get_batch(self, idx: int, batch_size: int):
        """Get batch data."""
        batch_samples = self.samples[idx:idx + batch_size]

        image_filenames = [s['image_filename'] for s in batch_samples]
        cands = [s['candidate'] for s in batch_samples]
        refs = [s['references'] if self.use_references else [] for s in batch_samples]
        scores = [s['score'] for s in batch_samples]

        image_paths = [str(self.img_dir / fname) for fname in image_filenames]

        valid_indices = [i for i, p in enumerate(image_paths) if os.path.exists(p)]

        if not valid_indices:
            return None

        return {
            'image_ids': [image_filenames[i] for i in valid_indices],
            'image_paths': [image_paths[i] for i in valid_indices],
            'candidates': [cands[i] for i in valid_indices],
            'references': [refs[i] for i in valid_indices],
            'scores': [scores[i] for i in valid_indices]
        }


def evaluate_flickr8k(model, dataset: Flickr8kDataset, device: torch.device,
                     batch_size: int = 8) -> Dict:
    """Evaluate on Flickr8k dataset."""
    print("\n" + "=" * 80)
    ref_mode = "with references" if dataset.use_references else "without references"
    print(f"Flickr8k {dataset.variant.upper()} Evaluation ({ref_mode})")
    print("=" * 80)

    predictions = []
    ground_truth = []

    with torch.no_grad():
        for idx in tqdm(range(0, len(dataset), batch_size), desc=f"Flickr8k {dataset.variant} evaluation"):
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

    correlations = compute_correlations(predictions, ground_truth, dataset.kendall_variant)

    primary_metric = f"kendall_tau_{dataset.kendall_variant}"
    print("\nResults:")
    print(f"  Kendall Tau-b: {correlations['kendall_tau_b']:.2f}")
    print(f"  Kendall Tau-c: {correlations['kendall_tau_c']:.2f}")
    print(f"  Primary metric: Kendall Tau-{dataset.kendall_variant.upper()} = "
          f"{correlations[primary_metric]:.2f}")

    return {
        "variant": dataset.variant,
        "num_samples": len(predictions),
        "kendall_variant": dataset.kendall_variant,
        "correlations": correlations,
        "predictions_mean": float(predictions.mean()),
        "predictions_std": float(predictions.std()),
        "ground_truth_mean": float(ground_truth.mean()),
        "ground_truth_std": float(ground_truth.std()),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Flickr8k-CF & Expert Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Local weights (.safetensors/.pth). Defaults to the Hugging Face release")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Local config YAML. Defaults to the Hugging Face release")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing Flickr8k data")
    parser.add_argument("--output_dir", type=str, default="results/flickr8k",
                       help="Output directory for results")
    parser.add_argument("--variant", type=str, choices=["expert", "cf"],
                       help="Evaluation variant (expert or cf)")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--no_references", action="store_true",
                       help="Evaluate without references")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate both Expert and CF variants")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    checkpoint_path, config_path = resolve_checkpoint(args.checkpoint_path, args.config_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, config_path, device)

    if args.all:
        variants = ["expert", "cf"]
    elif args.variant:
        variants = [args.variant]
    else:
        raise ValueError("Please specify --variant or --all")

    all_results = {}

    for variant in variants:
        dataset = Flickr8kDataset(data_root, variant=variant, use_references=not args.no_references)
        results = evaluate_flickr8k(model, dataset, device, args.batch_size)
        all_results[variant] = results

        output_file = output_dir / f"flickr8k_{variant}_{'ref' if not args.no_references else 'noref'}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "variant": variant,
                    "use_references": not args.no_references,
                    "batch_size": args.batch_size
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        for variant, results in all_results.items():
            primary = f"kendall_tau_{results['kendall_variant']}"
            print(f"{variant.upper()}: {results['correlations'][primary]:.2f}")


if __name__ == "__main__":
    main()
