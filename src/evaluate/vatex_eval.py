"""VATEX-EVAL Evaluation Script

Uses the official VATEX-EVAL dataset (JSON file) and computes correlation
by averaging correlations with 3 human evaluators (official method).

Usage:
    uv run python src/evaluate/vatex_eval.py \
        --json_path data/vatex_eval/vid_cans_score_dict_available.json \
        --video_dir data/vatex_eval/videos \
        --output_dir results/vatex_eval \
        --use_n_refs 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src/ to path so the scripts also run without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rigel import QwenLoRAScoreSmoothing, load_checkpoint_strict, resolve_checkpoint


def load_model(checkpoint_path: Path, config_path: Path, device: torch.device):
    """Load the Rigel model for video evaluation."""
    cfg = OmegaConf.load(str(config_path))

    model = QwenLoRAScoreSmoothing(cfg)
    load_checkpoint_strict(model, checkpoint_path, map_location=device, verbose=True)
    model.eval()
    model.to(device)
    return model, cfg


def compute_correlation_uniquehuman(pred, all_human_scores):
    """Official method: average correlations with 3 human evaluators."""
    num_workers = 3

    pred = np.around(pred, decimals=4)

    spearman = 0
    for worker_i in range(num_workers):
        tmp, p_value = spearmanr(pred, all_human_scores[:, worker_i])
        if p_value >= 0.01:
            print(f"  ⚠ Warning: p-value {p_value:.4f} >= 0.01 for worker {worker_i+1}")
        else:
            print(f"  ✓ Worker {worker_i+1}: Spearman {tmp:.4f} (p-value: {p_value:.4e})")
        spearman += tmp
    spearman /= num_workers
    spearman = np.around(spearman, decimals=4)

    kendalltau_val = 0
    for worker_i in range(num_workers):
        tmp, p_value = kendalltau(pred, all_human_scores[:, worker_i])
        if p_value >= 0.01:
            print(f"  ⚠ Warning: p-value {p_value:.4f} >= 0.01 for worker {worker_i+1}")
        else:
            print(f"  ✓ Worker {worker_i+1}: Kendall {tmp:.4f} (p-value: {p_value:.4e})")
        kendalltau_val += tmp
    kendalltau_val /= num_workers
    kendalltau_val = np.around(kendalltau_val, decimals=4)

    print('\nFinal Results (average of 3 evaluators):')
    print(f'  Kendall Tau:  {kendalltau_val:.4f}')
    print(f'  Spearman Rho: {spearman:.4f}')

    return kendalltau_val, spearman


def evaluate_official(
    model,
    candidates,
    gts,
    video_ids,
    all_human_scores,
    video_dir: Path,
    device: torch.device,
    use_n_refs: int = 1,
    debug_limit: Optional[int] = None
):
    """Evaluate on official dataset."""
    print("=" * 80)
    print("Evaluating on Official VATEX Eval Dataset")
    print("=" * 80)
    print(f"Number of samples: {len(candidates)}")
    print(f"Number of references: {use_n_refs}")
    print(f"Number of evaluators: {all_human_scores.shape[1]}")

    if debug_limit:
        candidates = candidates[:debug_limit]
        gts = gts[:debug_limit]
        video_ids = video_ids[:debug_limit]
        all_human_scores = all_human_scores[:debug_limit]
        print(f"Debug mode: evaluating only first {debug_limit} samples")

    refs = gts[:, :use_n_refs].tolist()
    predictions = []

    print("\nRunning inference...")

    with torch.no_grad():
        for idx in tqdm(range(len(candidates)), desc="Evaluating"):
            video_id = video_ids[idx]
            video_path = video_dir / f"{video_id}.mp4"

            if not video_path.exists():
                predictions.append(0.5)
                continue

            cand = candidates[idx]
            ref = refs[idx]

            try:
                pred_score = model(
                    media_path=[str(video_path)],
                    cand=[cand],
                    refs=[ref],
                    media_id=[video_id],
                    modality=["video"]
                )

                pred_score_value = pred_score.item() if torch.is_tensor(pred_score) else pred_score[0]
                predictions.append(pred_score_value)

            except Exception:
                predictions.append(0.5)
                continue

    predictions = np.array(predictions)

    print("\n" + "=" * 80)
    print("Computing Correlations (Official Method: Average of 3 Evaluators)")
    print("=" * 80)
    kendall_tau, spearman_rho = compute_correlation_uniquehuman(predictions, all_human_scores)

    return {
        'kendall_tau': float(kendall_tau),
        'spearman_rho': float(spearman_rho),
        'num_samples': len(predictions),
        'use_n_refs': use_n_refs,
        'predictions': predictions.tolist(),
        'pred_mean': float(np.mean(predictions)),
        'pred_std': float(np.std(predictions)),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="VATEX-EVAL Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Local weights (.safetensors/.pth). Defaults to the Hugging Face release")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Local config YAML. Defaults to the Hugging Face release")
    parser.add_argument("--json_path", type=str, required=True,
                       help="Path to official JSON file (vid_cans_score_dict_available.json)")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video files")
    parser.add_argument("--output_dir", type=str, default="results/vatex_eval",
                       help="Output directory for results")
    parser.add_argument("--use_n_refs", type=int, default=1,
                       help="Number of references to use (1~9)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--debug_limit", type=int, default=None,
                       help="Debug mode: maximum number of samples")

    args = parser.parse_args()

    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please specify --device cpu")
    device = torch.device(device_str)

    checkpoint_path, config_path = resolve_checkpoint(args.checkpoint_path, args.config_path)
    json_path = Path(args.json_path).expanduser().resolve()
    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("VATEX-EVAL Evaluation")
    print("=" * 80)
    print(f"Config file:     {config_path}")
    if checkpoint_path:
        print(f"Checkpoint:       {checkpoint_path}")
    print(f"JSON file:        {json_path}")
    print(f"Video directory:  {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device:           {device}")
    print(f"Number of refs:   {args.use_n_refs}")
    if args.debug_limit:
        print(f"Debug limit:      {args.debug_limit} samples")
    print()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    print(f"Loaded {json_path.name}: {len(json_data)} videos")

    candidates = []
    gts_list = []
    video_ids_list = []
    scores_list = []

    for vid, data in json_data.items():
        cands = data['cands']
        refs = data['refs'][0]
        scores = eval(data['scores'])

        for cand_idx, cand in enumerate(cands):
            candidates.append(cand)
            video_ids_list.append(vid)

            padded_refs = refs + [''] * (9 - len(refs))
            gts_list.append(padded_refs[:9])

            scores_list.append(scores[cand_idx])

    candidates = np.array(candidates)
    gts = np.array(gts_list)
    video_ids = np.array(video_ids_list)
    all_human_scores = np.array(scores_list)

    print(f"  Candidate captions: {len(candidates)}")
    print(f"  References: {gts.shape}")
    print(f"  Human scores: {all_human_scores.shape}")
    print(f"  Video IDs: {len(video_ids)}")
    print(f"\n✓ Video directory: {video_dir}")
    print(f"  Exists: {'✓' if video_dir.exists() else '✗'}")

    model, cfg = load_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)

    results = evaluate_official(
        model=model,
        candidates=candidates,
        gts=gts,
        video_ids=video_ids,
        all_human_scores=all_human_scores,
        video_dir=video_dir,
        device=device,
        use_n_refs=args.use_n_refs,
        debug_limit=args.debug_limit
    )

    output_file = output_dir / f"vatex_eval_{args.use_n_refs}ref.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "config": {
                "checkpoint_path": str(checkpoint_path),
                "config_path": str(config_path),
                "model_class": cfg.get("model_class"),
                "use_n_refs": args.use_n_refs,
                "debug_limit": args.debug_limit
            },
            "results": {
                k: v for k, v in results.items() if k != "predictions"
            }
        }, f, indent=2, ensure_ascii=False)

    detailed_output_file = output_dir / f"vatex_eval_{args.use_n_refs}ref_detailed.json"
    with open(detailed_output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("Evaluation Complete")
    print("=" * 80)
    print(f"\nFinal Results ({args.use_n_refs}ref):")
    print(f"  Kendall Tau:  {results['kendall_tau']:.4f}")
    print(f"  Spearman Rho: {results['spearman_rho']:.4f}")
    print(f"  Number of samples: {results['num_samples']}")

    print("\n✓ Results saved:")
    print(f"  Summary: {output_file}")
    print(f"  Detailed: {detailed_output_file}")


if __name__ == "__main__":
    main()
