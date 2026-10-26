"""FOIL Dataset Evaluation Script

Usage:
    # (one reference)
    uv run python src/evaluate/foil.py \
        --data_root data/foil \
        --output_dir results/foil \
        --one_ref

    # (four references)
    uv run python src/evaluate/foil.py \
        --data_root data/foil \
        --output_dir results/foil \
        --four_ref

    # All variants at once
    uv run python src/evaluate/foil.py \
        --data_root data/foil \
        --output_dir results/foil \
        --all
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf
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


class FoilDataset:
    """FOIL dataset loader."""

    def __init__(self, data_root: Path):
        """Initialize FOIL dataset."""
        self.data_root = Path(data_root)

        self.foil_json = self.data_root / "foilv1.0_test_2017.json"
        self.ref_json = self.data_root / "captions_val2014.json"
        self.images_dir = self.data_root / "images"

        if not self.foil_json.exists():
            raise FileNotFoundError(f"FOIL JSON not found: {self.foil_json}")
        if not self.ref_json.exists():
            raise FileNotFoundError(f"Reference JSON not found: {self.ref_json}")
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")

        coco_refs = self._read_coco(self.ref_json)
        self.data = self._build_foil(self.foil_json, coco_refs)
        self.dataset = {"one_ref": None, "four_ref": None}

        print("✓ FOIL dataset loaded")
        print(f"  anno_id count: {len(self.data)}")

    def _read_coco(self, coco_annos: Path) -> Dict[int, List[str]]:
        """Load COCO references."""
        refs = {}
        with open(coco_annos) as f:
            coco = json.load(f)
        for ann in coco["annotations"]:
            refs.setdefault(ann['image_id'], []).append(ann['caption'])
        return refs

    def _build_foil(self, path: Path, coco_refs: Dict[int, List[str]]) -> Dict:
        """Build FOIL data structure."""
        with open(path) as f:
            foil_data = json.load(f)

        images = foil_data["images"]
        annos = foil_data["annotations"]

        data = {}
        imgid_to_img = {img["id"]: img for img in images}

        for anno in annos:
            anno_id = anno["id"]
            data.setdefault(anno_id, {"foil": [], "orig": []})

            key = "foil" if anno["foil"] else "orig"
            anno["image"] = imgid_to_img[anno["image_id"]]
            anno["refs"] = coco_refs.get(anno["image_id"], [])

            data[anno_id][key].append(anno)

        return data

    def get_data(self, one_ref: bool) -> List[Dict]:
        """Get data for evaluation. Data is ordered as foil, orig, foil, orig, ..."""
        key = "one_ref" if one_ref else "four_ref"
        if self.dataset[key] is not None:
            return self.dataset[key]

        dataset = []

        for _, data_item in (pbar := tqdm(self.data.items())):
            pbar.set_description("Prepare dataset ...")

            foiles, origs = data_item["foil"], data_item["orig"]

            assert len(origs) == 1, f"orig count is not 1: {len(origs)}"

            N = len(foiles)
            for foil, orig in zip(foiles, [origs[0]] * N):
                refs = foil["refs"]
                refs = [r for r in refs if r != orig["caption"]]

                if one_ref:
                    refs = [refs[0]] if refs else []
                else:
                    refs = refs[:4]

                filename = Path(foil["image"]["file_name"])
                img_path = self.images_dir / filename
                dataset.append({
                    "imgid": str(img_path),
                    "refs": refs,
                    "mt": foil["caption"],
                    "type": "foil"
                })
                dataset.append({
                    "imgid": str(img_path),
                    "refs": refs,
                    "mt": orig["caption"],
                    "type": "orig"
                })

        self.dataset[key] = dataset
        return self.dataset[key]


def compute_acc(model, dataset: FoilDataset, one_ref: bool, device: torch.device,
                batch_size: int = 8) -> float:
    """Compute accuracy (proportion of orig_score > foil_score)."""
    bucket_count = 20
    data = dataset.get_data(one_ref)

    print("Compute ...")
    sys_score = []

    for i in range(bucket_count):
        bucket_size = len(data) // bucket_count
        subset = deepcopy(data[i * bucket_size:(i + 1) * bucket_size])

        with torch.no_grad():
            sub_sys_score = []

            for idx in tqdm(range(0, len(subset), batch_size),
                          desc=f"Processing {i+1}/{bucket_count}",
                          leave=False):
                batch = subset[idx:idx + batch_size]

                image_paths = [item['imgid'] for item in batch]
                candidates = [item['mt'] for item in batch]
                references = [item['refs'] for item in batch]

                valid_indices = [j for j, path in enumerate(image_paths) if os.path.exists(path)]
                if not valid_indices:
                    sub_sys_score.extend([0.5] * len(batch))
                    continue

                valid_image_paths = [image_paths[j] for j in valid_indices]
                valid_candidates = [candidates[j] for j in valid_indices]
                valid_references = [references[j] for j in valid_indices]
                valid_image_ids = [os.path.basename(path) for path in valid_image_paths]

                try:
                    pred_scores = model(
                        media_id=valid_image_ids,
                        media_path=valid_image_paths,
                        cand=valid_candidates,
                        refs=valid_references,
                        modality=["image"] * len(valid_image_ids)
                    )

                    if torch.is_tensor(pred_scores):
                        pred_scores = pred_scores.cpu().numpy()
                    elif isinstance(pred_scores, (list, tuple)):
                        pred_scores = np.array(pred_scores)
                    else:
                        pred_scores = np.array([float(pred_scores)])

                    batch_scores = []
                    valid_idx = 0
                    for j in range(len(batch)):
                        if j in valid_indices:
                            batch_scores.append(float(pred_scores[valid_idx]))
                            valid_idx += 1
                        else:
                            batch_scores.append(0.5)

                    sub_sys_score.extend(batch_scores)

                except Exception as e:
                    print(f"  ⚠ Inference error: {e}")
                    sub_sys_score.extend([0.5] * len(batch))
                    continue

            sys_score.extend(sub_sys_score)

        del subset

    assert len(sys_score) == len(data), f"Score count mismatch: {len(sys_score)} != {len(data)}"
    assert len(sys_score) % 2 == 0, f"Score count is not even: {len(sys_score)}"

    acc = 0.
    N = len(sys_score) // 2

    for i in range(0, 2 * N, 2):
        s1 = sys_score[i]  # foil
        s2 = sys_score[i + 1]  # orig

        assert data[i]["type"] == "foil" and data[i + 1]["type"] == "orig", \
            f"Data order mismatch: {data[i]['type']}, {data[i+1]['type']}"

        if s2 > s1:
            acc += 1.

    acc /= N

    print(f"acc: {acc:.4f}")

    return acc


def compute_foil(model, dataset: FoilDataset, device: torch.device,
                 batch_size: int = 8) -> Dict:
    """Run FOIL evaluation for both one_ref and four_ref."""
    memory = {}
    dataset_name = "foil"

    for one_ref in [True, False]:
        suffix = "(one_ref)" if one_ref else "(four-ref)"
        full_dataset_name = dataset_name + suffix

        acc = compute_acc(model, dataset, one_ref, device, batch_size)
        memory[full_dataset_name] = {"Rigel" + suffix: acc}

    all_results = {}
    for dataset_name_key, methods in memory.items():
        for method, acc in methods.items():
            all_results[method] = acc

    return {
        "memory": memory,
        "results": all_results,
        "one_ref_accuracy": memory.get("foil(one_ref)", {}).get("Rigel(one_ref)", 0.0),
        "four_ref_accuracy": memory.get("foil(four-ref)", {}).get("Rigel(four-ref)", 0.0),
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="FOIL Dataset Evaluation")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Local weights (.safetensors/.pth). Defaults to the Hugging Face release")
    parser.add_argument("--config_path", type=str, default=None,
                       help="Local config YAML. Defaults to the Hugging Face release")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory containing FOIL data")
    parser.add_argument("--output_dir", type=str, default="results/foil",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--one_ref", action="store_true",
                       help="Evaluate with one reference")
    parser.add_argument("--four_ref", action="store_true",
                       help="Evaluate with four references")
    parser.add_argument("--all", action="store_true",
                       help="Evaluate both one_ref and four_ref")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    checkpoint_path, config_path = resolve_checkpoint(args.checkpoint_path, args.config_path)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, config_path, device)
    dataset = FoilDataset(data_root)

    if args.all:
        results = compute_foil(model, dataset, device, args.batch_size)

        output_file = output_dir / "foil_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "batch_size": args.batch_size
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("Final Results")
        print("=" * 80)
        print(f"1-ref accuracy: {results['one_ref_accuracy']:.4f}")
        print(f"4-ref accuracy: {results['four_ref_accuracy']:.4f}")
        print(f"\n✓ Results saved to: {output_file}")

    else:
        if args.one_ref:
            acc = compute_acc(model, dataset, one_ref=True, device=device, batch_size=args.batch_size)
            results = {"one_ref_accuracy": acc}
        elif args.four_ref:
            acc = compute_acc(model, dataset, one_ref=False, device=device, batch_size=args.batch_size)
            results = {"four_ref_accuracy": acc}
        else:
            raise ValueError("Please specify --one_ref, --four_ref, or --all")

        ref_mode_name = "one_ref" if args.one_ref else "four_ref"
        output_file = output_dir / f"foil_{ref_mode_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "config": {
                    "checkpoint_path": str(checkpoint_path),
                    "config_path": str(config_path),
                    "use_one_ref": args.one_ref,
                    "batch_size": args.batch_size
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()
