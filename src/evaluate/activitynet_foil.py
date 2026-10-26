"""ActivityNet-FOIL Evaluation Script

Loads Rigel model checkpoint and configuration, then evaluates by comparing
Right / Foil caption JSON files from ActivityNet-FOIL.

Default JSON files:
    - final_right_video_sentences.json (Right predictions)
    - final_foil_video_sentences.json  (Foil predictions)
    - anet_entities_test_2.json        (References)

Usage:
    # With references
    uv run python src/evaluate/activitynet_foil.py \
        --storage_path data/anet_foil \
        --output_dir results/activitynet_foil \
        --use_references

    # Without references
    uv run python src/evaluate/activitynet_foil.py \
        --storage_path data/anet_foil \
        --output_dir results/activitynet_foil_noref \
        --no_references
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.io import read_video
from tqdm import tqdm

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False

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


@dataclass
class EvaluationSample:
    video_id: str
    segment_index: int
    timestamp: Optional[Sequence[float]]
    right_sentence: str
    foil_sentence: str
    right_score: float
    foil_score: float


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_reference_sentences(
    timestamp: Optional[Sequence[float]],
    ref_data: Optional[Dict],
    use_references: bool,
    overlap_threshold: float,
) -> List[str]:
    if not use_references or not ref_data:
        return []

    sentences = ref_data.get("sentences", [])
    ref_timestamps = ref_data.get("timestamps", [])

    if not sentences:
        return []

    if not timestamp or not ref_timestamps or len(ref_timestamps) != len(sentences):
        return sentences

    try:
        start, end = float(timestamp[0]), float(timestamp[1])
    except (TypeError, ValueError, IndexError):
        return sentences

    if start >= end:
        return sentences

    start_int = int(start)
    end_int = int(end)
    if start_int > end_int:
        start_int, end_int = end_int, start_int

    base_range = set(range(start_int, end_int + 1))
    if not base_range:
        return sentences

    selected_indices = []
    for idx, ref_time in enumerate(ref_timestamps):
        try:
            ref_start, ref_end = float(ref_time[0]), float(ref_time[1])
        except (TypeError, ValueError, IndexError):
            continue

        ref_start_int = int(ref_start)
        ref_end_int = int(ref_end)
        if ref_start_int > ref_end_int:
            ref_start_int, ref_end_int = ref_end_int, ref_start_int

        ref_range = list(range(ref_start_int, ref_end_int + 1))
        if not ref_range:
            continue

        intersection = base_range.intersection(ref_range)
        if len(intersection) / len(ref_range) > overlap_threshold:
            selected_indices.append(idx)

    if not selected_indices:
        return sentences

    return [sentences[i] for i in selected_indices]


def score_caption(
    model,
    video_id: Sequence[str] | str,
    video_path: Sequence[Path] | Sequence[str] | Path,
    candidate: Sequence[str] | str,
    references: Sequence[List[str]] | List[str],
):
    single = isinstance(video_id, str)

    if single:
        video_ids = [video_id]
        video_paths = [str(video_path)] if isinstance(video_path, (str, Path)) else [str(video_path[0])]
        candidates = [candidate] if isinstance(candidate, str) else list(candidate)
        if isinstance(references, list) and (not references or isinstance(references[0], str)):
            refs = [list(references)]
        else:
            refs = [list(references)]
    else:
        video_ids = list(video_id)
        if isinstance(video_path, (list, tuple)):
            video_paths = [str(p) for p in video_path]
        else:
            raise ValueError("video_path must be a sequence when batching multiple samples")
        candidates = list(candidate)
        refs = []
        for item in references:
            if isinstance(item, (list, tuple)):
                refs.append(list(item))
            else:
                refs.append([])

    with torch.inference_mode():
        output = model(
            media_id=video_ids,
            media_path=video_paths,
            cand=candidates,
            refs=refs,
            modality=["video"] * len(video_ids)
        )

    scores: List[float] = []
    if isinstance(output, torch.Tensor):
        scores = output.detach().cpu().flatten().tolist()
    elif isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                scores.extend(item.detach().cpu().flatten().tolist())
            else:
                try:
                    scores.append(float(item))
                except (TypeError, ValueError):
                    continue
    else:
        try:
            scores = [float(output)]
        except (TypeError, ValueError):
            scores = list(output)

    if single:
        return float(scores[0]) if scores else 0.0
    return scores


def extract_segment_video(
    video_path: Path,
    timestamp: Optional[Sequence[float]],
    cache: Dict[Tuple[str, int, int], Path],
    segment_dir: Path,
) -> Path:
    if not timestamp or len(timestamp) < 2:
        return video_path

    try:
        start = float(timestamp[0])
        end = float(timestamp[1])
    except (TypeError, ValueError):
        return video_path

    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        return video_path

    key = (video_path.as_posix(), int(round(start * 1000)), int(round(end * 1000)))
    cached = cache.get(key)
    if cached and cached.exists():
        return cached

    seg_filename = f"{video_path.stem}_{key[1]}_{key[2]}.mp4"
    seg_path = segment_dir / seg_filename
    if seg_path.exists():
        cache[key] = seg_path
        return seg_path
    try:
        video_frames, _, info = read_video(
            str(video_path),
            start_pts=start,
            end_pts=end,
            pts_unit="sec",
        )
    except Exception:  # pylint: disable=broad-except
        return video_path

    if video_frames.numel() == 0:
        return video_path

    fps = info.get("video_fps") or 30
    fps = float(fps)

    try:
        # Use PyAV directly to avoid torchvision write_video issues
        if not HAS_AV:
            return video_path

        # Ensure video_frames is uint8 and in correct format (T, H, W, C)
        if video_frames.dtype != torch.uint8:
            video_frames = video_frames.to(torch.uint8)

        # Convert to numpy for PyAV
        frames_np = video_frames.cpu().numpy()

        # Write video using PyAV directly
        container = av.open(str(seg_path), mode='w')
        # Convert fps to Fraction for PyAV
        fps_fraction = Fraction(fps).limit_denominator(1000)
        stream = container.add_stream('h264', rate=fps_fraction)
        stream.width = frames_np.shape[2]
        stream.height = frames_np.shape[1]
        stream.pix_fmt = 'yuv420p'

        for frame_np in frames_np:
            frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush remaining packets
        for packet in stream.encode():
            container.mux(packet)

        container.close()
    except Exception:  # pylint: disable=broad-except
        return video_path

    cache[key] = seg_path
    return seg_path


def evaluate_predictions(
    model,
    right_predictions: Dict[str, List[Dict]],
    foil_predictions: Dict[str, List[Dict]],
    references: Dict[str, Dict],
    candidate_metadata: Dict[str, Dict],
    video_dir: Path,
    segment_cache_dir: Path,
    use_references: bool,
    overlap_threshold: float,
) -> Dict[str, object]:
    samples: List[EvaluationSample] = []
    skipped_missing_video: List[str] = []
    skipped_mismatch: List[str] = []
    skipped_missing_candidate: List[str] = []
    inference_errors: List[Dict[str, object]] = []

    shared_video_ids = sorted(set(right_predictions.keys()) & set(foil_predictions.keys()))

    segment_cache_dir.mkdir(parents=True, exist_ok=True)
    segment_cache: Dict[Tuple[str, int, int], Path] = {}

    video_level_wins = 0
    total_videos_processed = 0
    debug_count = 0
    debug_limit = 50

    current_video_right_scores = []
    current_video_foil_scores = []

    pbar = tqdm(shared_video_ids, desc="Computing Rigel scores")
    for vid in pbar:
            current_video_right_scores = []
            current_video_foil_scores = []

            video_path = video_dir / f"{vid}.mp4"
            if not video_path.exists():
                skipped_missing_video.append(vid)
                continue

            right_segments = right_predictions.get(vid, [])
            foil_segments = foil_predictions.get(vid, [])
            if not right_segments or not foil_segments:
                skipped_mismatch.append(vid)
                continue

            candidate_info = candidate_metadata.get(vid)
            if candidate_info is None:
                skipped_missing_candidate.append(vid)
                continue

            candidate_timestamps = (
                candidate_info.get("timestamps") if isinstance(candidate_info, dict) else None
            )
            if not candidate_timestamps:
                skipped_missing_candidate.append(vid)
                continue

            if len(right_segments) != len(foil_segments) or len(right_segments) != len(candidate_timestamps):
                skipped_mismatch.append(vid)

            paired_len = min(len(right_segments), len(foil_segments), len(candidate_timestamps))
            for idx in range(paired_len):
                right_entry = right_segments[idx]
                foil_entry = foil_segments[idx]

                timestamp = right_entry.get("timestamp") or foil_entry.get("timestamp")
                timestamp_missing = not timestamp or not isinstance(timestamp, (list, tuple))
                if timestamp_missing and idx < len(candidate_timestamps):
                    timestamp = candidate_timestamps[idx]

                segment_video_path = extract_segment_video(
                    video_path=video_path,
                    timestamp=timestamp,
                    cache=segment_cache,
                    segment_dir=segment_cache_dir,
                )

                ref_sentences = select_reference_sentences(
                    timestamp=timestamp,
                    ref_data=references.get(vid),
                    use_references=use_references,
                    overlap_threshold=overlap_threshold,
                )

                if debug_count < debug_limit:
                    print(f"\n[DEBUG] Sample {debug_count + 1}")
                    print(f"  Video: {vid}, Segment: {idx}")
                    print(f"  Timestamp: {timestamp}")
                    print(f"  Original video: {video_path}")
                    print(f"  Segment video path: {segment_video_path}")
                    print(f"  Is segment (not original): {segment_video_path != video_path}")
                    print(f"  Segment exists: {segment_video_path.exists()}")
                    if segment_video_path.exists():
                        from torchvision.io import read_video as dbg_read_video
                        try:
                            dbg_frames, _, dbg_info = dbg_read_video(str(segment_video_path), pts_unit="sec")
                            dbg_duration = dbg_frames.shape[0] / dbg_info.get('video_fps', 30)
                            print(f"  Segment duration: {dbg_duration:.2f}s ({dbg_frames.shape[0]} frames)")
                        except Exception:
                            pass
                    print(f"  Right: {right_entry.get('sentence', '')[:80]}...")
                    print(f"  Foil:  {foil_entry.get('sentence', '')[:80]}...")
                    print(f"  Refs: {len(ref_sentences)} sentences")

                try:
                    scores = score_caption(
                        model=model,
                        video_id=[vid, vid],
                        video_path=[segment_video_path, segment_video_path],
                        candidate=[right_entry.get("sentence", ""), foil_entry.get("sentence", "")],
                        references=[ref_sentences, ref_sentences],
                    )

                    if debug_count < debug_limit:
                        is_pair = isinstance(scores, (list, tuple))
                        right_dbg = scores[0] if is_pair else scores
                        foil_dbg = scores[1] if is_pair and len(scores) > 1 else scores
                        print(f"  Scores: Right={right_dbg:.4f}, Foil={foil_dbg:.4f}")
                        debug_count += 1

                except Exception as exc:  # pylint: disable=broad-except
                    inference_errors.append(
                        {
                            "video_id": vid,
                            "segment_index": idx,
                            "error": repr(exc),
                        }
                    )
                    continue

                if isinstance(scores, (list, tuple)) and len(scores) >= 2:
                    right_score, foil_score = float(scores[0]), float(scores[1])
                elif isinstance(scores, np.ndarray) and scores.size >= 2:
                    right_score, foil_score = float(scores[0]), float(scores[1])
                else:
                    val = float(scores) if not isinstance(scores, (list, tuple)) else float(scores[0])
                    right_score = val
                    foil_score = val

                samples.append(
                    EvaluationSample(
                        video_id=vid,
                        segment_index=idx,
                        timestamp=timestamp,
                        right_sentence=right_entry.get("sentence", ""),
                        foil_sentence=foil_entry.get("sentence", ""),
                        right_score=right_score,
                        foil_score=foil_score,
                    )
                )

                current_video_right_scores.append(right_score)
                current_video_foil_scores.append(foil_score)

            if current_video_right_scores and current_video_foil_scores:
                mean_right = np.mean(current_video_right_scores)
                mean_foil = np.mean(current_video_foil_scores)

                if mean_right > mean_foil:
                    video_level_wins += 1

                total_videos_processed += 1

                video_acc = (video_level_wins / total_videos_processed) * 100
                pbar.set_postfix({
                    "VideoAcc": f"{video_acc:.2f}%",
                    "Videos": f"{video_level_wins}/{total_videos_processed}"
                })

    if not samples:
        return {
            "samples": [],
            "metrics": {},
            "skipped_missing_video": skipped_missing_video,
            "skipped_mismatch": skipped_mismatch,
            "skipped_missing_candidate": skipped_missing_candidate,
            "inference_errors": inference_errors,
        }

    right_scores = np.array([sample.right_score for sample in samples], dtype=np.float32)
    foil_scores = np.array([sample.foil_score for sample in samples], dtype=np.float32)

    metrics = {
        "num_pairs": int(len(samples)),
        "num_videos": int(len({s.video_id for s in samples})),
        "right_mean": float(right_scores.mean()),
        "right_std": float(right_scores.std()),
        "foil_mean": float(foil_scores.mean()),
        "foil_std": float(foil_scores.std()),
    }

    samples_dict = [
        {
            "video_id": s.video_id,
            "segment_index": s.segment_index,
            "timestamp": list(s.timestamp) if isinstance(s.timestamp, (list, tuple)) else s.timestamp,
            "right_sentence": s.right_sentence,
            "foil_sentence": s.foil_sentence,
            "right_score": float(s.right_score),
            "foil_score": float(s.foil_score),
            "margin": float(s.right_score - s.foil_score),
        }
        for s in samples
    ]

    video_grouped: Dict[str, List[EvaluationSample]] = {}
    for sample in samples:
        video_grouped.setdefault(sample.video_id, []).append(sample)

    video_metrics = {}
    right_video_means = []
    foil_video_means = []
    for vid, segs in video_grouped.items():
        right_arr = np.array([seg.right_score for seg in segs], dtype=np.float32)
        foil_arr = np.array([seg.foil_score for seg in segs], dtype=np.float32)
        right_mean = float(right_arr.mean())
        foil_mean = float(foil_arr.mean())
        video_metrics[vid] = {
            "num_segments": len(segs),
            "right_mean": right_mean,
            "foil_mean": foil_mean,
        }
        right_video_means.append(right_mean)
        foil_video_means.append(foil_mean)

    video_level_accuracy = {}
    if right_video_means and len(right_video_means) == len(foil_video_means):
        right_arr = np.array(right_video_means, dtype=np.float32)
        foil_arr = np.array(foil_video_means, dtype=np.float32)
        wins = right_arr > foil_arr
        win_count = int(wins.sum())
        total = len(right_arr)
        accuracy = float(win_count * 100.0 / max(total, 1))
        video_level_accuracy = {
            "correct_videos": win_count,
            "total_videos": total,
            "accuracy": accuracy,
        }

    return {
        "samples": samples_dict,
        "metrics": metrics,
        "skipped_missing_video": skipped_missing_video,
        "skipped_mismatch": skipped_mismatch,
        "skipped_missing_candidate": skipped_missing_candidate,
        "inference_errors": inference_errors,
        "video_metrics": video_metrics,
        "video_level_accuracy": video_level_accuracy,
    }


def main(args: argparse.Namespace) -> None:
    storage_path = Path(args.storage_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please specify --device cpu")
    device = torch.device(device_str)

    checkpoint_path, config_path = resolve_checkpoint(args.checkpoint_path, args.config_path)

    right_predictions_path = (
        Path(args.right_predictions).expanduser().resolve()
        if args.right_predictions
        else storage_path / "final_right_video_sentences.json"
    )
    foil_predictions_path = (
        Path(args.foil_predictions).expanduser().resolve()
        if args.foil_predictions
        else storage_path / "final_foil_video_sentences.json"
    )
    references_path = (
        Path(args.references_path).expanduser().resolve()
        if args.references_path
        else storage_path / "anet_entities_test_2.json"
    )
    candidate_timestamps_path = (
        Path(args.candidate_timestamps).expanduser().resolve()
        if args.candidate_timestamps
        else storage_path / "anet_entities_test_1.json"
    )
    video_dir = (
        Path(args.video_dir).expanduser().resolve()
        if args.video_dir
        else storage_path / "videos"
    )

    if not right_predictions_path.exists():
        raise FileNotFoundError(f"Right predictions JSON not found: {right_predictions_path}")
    if not foil_predictions_path.exists():
        raise FileNotFoundError(f"Foil predictions JSON not found: {foil_predictions_path}")
    if not references_path.exists():
        raise FileNotFoundError(f"References JSON not found: {references_path}")
    if not candidate_timestamps_path.exists():
        raise FileNotFoundError(f"Candidate timestamps JSON not found: {candidate_timestamps_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    model, _cfg = load_model(checkpoint_path=checkpoint_path, config_path=config_path, device=device)

    right_predictions = read_json(right_predictions_path).get("results", {})
    foil_predictions = read_json(foil_predictions_path).get("results", {})
    references = read_json(references_path)
    candidate_metadata = read_json(candidate_timestamps_path)
    segment_cache_dir = (
        Path(args.segment_cache_dir).expanduser().resolve()
        if args.segment_cache_dir
        else storage_path / "cache" / "segments"
    )

    result = evaluate_predictions(
        model=model,
        right_predictions=right_predictions,
        foil_predictions=foil_predictions,
        references=references,
        candidate_metadata=candidate_metadata,
        video_dir=video_dir,
        segment_cache_dir=segment_cache_dir,
        use_references=args.use_references,
        overlap_threshold=args.overlap_threshold,
    )

    metrics = result.get("metrics", {})
    video_metrics = result.get("video_metrics", {})
    video_level_accuracy = result.get("video_level_accuracy", {})

    if metrics:
        print("\n" + "=" * 80)
        print("Evaluation Summary")
        print("=" * 80)
        print(f"Number of segments: {metrics['num_pairs']}")
        print(f"Number of videos:   {metrics['num_videos']}")
        print(f"Right mean score:   {metrics['right_mean']:.4f} ± {metrics['right_std']:.4f}")
        print(f"Foil mean score:    {metrics['foil_mean']:.4f} ± {metrics['foil_std']:.4f}")
    else:
        print("⚠ No samples were successfully evaluated")

    if video_level_accuracy:
        print("\n" + "=" * 80)
        print("Video-Level Accuracy (Official Evaluation)")
        print("=" * 80)
        correct = video_level_accuracy.get("correct_videos", 0)
        total = video_level_accuracy.get("total_videos", 0)
        accuracy = video_level_accuracy.get("accuracy", 0.0)
        print(f"Correct videos: {correct} / {total}")
        print(f"Accuracy:       {accuracy:.2f}%")
        print("=" * 80)

    skipped_info = {
        "missing_video": result.get("skipped_missing_video", []),
        "length_mismatch": result.get("skipped_mismatch", []),
        "missing_candidate": result.get("skipped_missing_candidate", []),
    }
    if skipped_info["missing_video"]:
        print(f"\nSkipped due to missing video: {len(skipped_info['missing_video'])}")
    if skipped_info["length_mismatch"]:
        print(f"Skipped due to caption count mismatch: {len(skipped_info['length_mismatch'])}")
    if skipped_info["missing_candidate"]:
        print(f"Skipped due to missing candidate timestamps: {len(skipped_info['missing_candidate'])}")

    if result.get("inference_errors"):
        print(f"Inference errors: {len(result['inference_errors'])} (details in json)")

    output_path = output_dir / "rigel_activitynet_foil_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": {
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "right_predictions_path": str(right_predictions_path),
                    "foil_predictions_path": str(foil_predictions_path),
                    "references_path": str(references_path),
                    "candidate_timestamps_path": str(candidate_timestamps_path),
                    "segment_cache_dir": str(segment_cache_dir),
                    "video_dir": str(video_dir),
                    "use_references": args.use_references,
                    "overlap_threshold": args.overlap_threshold,
                },
                "metrics": metrics,
                "video_level_accuracy": video_level_accuracy,
                "video_metrics": video_metrics,
                "samples": result.get("samples", []),
                "skipped_missing_video": result.get("skipped_missing_video", []),
                "skipped_mismatch": result.get("skipped_mismatch", []),
                "skipped_missing_candidate": result.get("skipped_missing_candidate", []),
                "inference_errors": result.get("inference_errors", []),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n✓ Evaluation results saved to: {output_path}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(description="ActivityNet-FOIL Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Local weights (.safetensors/.pth). Defaults to the Hugging Face release")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Local config YAML. Defaults to the Hugging Face release")
    parser.add_argument("--storage_path", type=str, default="data/anet_foil",
                        help="Root directory for ActivityNet-FOIL data")
    parser.add_argument("--right_predictions", type=str, default=None,
                        help="Right captions JSON (default: storage_path/final_right_video_sentences.json)")
    parser.add_argument("--foil_predictions", type=str, default=None,
                        help="Foil captions JSON (default: storage_path/final_foil_video_sentences.json)")
    parser.add_argument("--references_path", type=str, default=None,
                        help="References JSON (default: anet_entities_test_2.json)")
    parser.add_argument("--candidate_timestamps", type=str, default=None,
                        help="Candidate caption timestamps JSON (default: anet_entities_test_1.json)")
    parser.add_argument("--segment_cache_dir", type=str, default=None,
                        help="Video segment cache directory (default: storage_path/cache/segments)")
    parser.add_argument("--video_dir", type=str, default=None,
                        help="Video files directory (default: storage_path/videos)")
    parser.add_argument("--output_dir", type=str, default="results/activitynet_foil",
                        help="Output directory for results")
    parser.add_argument("--use_references", action="store_true", help="Include reference sentences in prompt")
    parser.add_argument("--no_references", dest="use_references", action="store_false",
                        help="Do not use reference sentences")
    parser.set_defaults(use_references=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.4,
                        help="Timestamp overlap threshold (0-1)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Inference device (e.g., cuda, cuda:1, cpu)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
