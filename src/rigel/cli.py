"""Command-line scorer: ``rigel-score --media video.mp4 --cand "..." [--ref "..."]``."""

import argparse
from typing import List, Optional

from .hub import DEFAULT_REPO_ID, from_pretrained


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a caption for a video or image with Rigel.")
    parser.add_argument("--media", type=str, required=True, help="Video or image file")
    parser.add_argument("--cand", type=str, nargs="+", required=True, help="Candidate caption(s)")
    parser.add_argument("--ref", type=str, action="append", default=[],
                        help="Reference caption (repeat for several; omit for reference-free)")
    parser.add_argument("--modality", choices=["video", "image"], default=None,
                        help="Defaults to a guess from the file extension")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID)
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Local weights (overrides Hub)")
    parser.add_argument("--config_path", type=str, default=None, help="Local config (overrides Hub)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    model = from_pretrained(
        repo_id=args.repo_id,
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        verbose=False,
    )
    n = len(args.cand)
    modality = [args.modality] * n if args.modality else None
    scores = model(media_path=[args.media] * n, cand=args.cand, refs=[list(args.ref)] * n, modality=modality)
    for cand, score in zip(args.cand, scores.tolist()):
        print(f"{score:.4f}\t{cand}")


if __name__ == "__main__":
    main()
