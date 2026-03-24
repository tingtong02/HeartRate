from __future__ import annotations

import argparse
from pathlib import Path

from heart_rate_cnn.results_site import build_results_site_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact JSON snapshots for the static results dashboard."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing outputs/ and web/.",
    )
    parser.add_argument(
        "--output-dir",
        default="web/public/data",
        help="Directory where the exported site JSON files should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_results_site_data(
        repo_root=Path(args.repo_root).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser(),
    )
    print("Results site data export completed.")
    print(f"Output directory: {summary['output_dir']}")
    print(f"Artifact count: {summary['artifact_count']}")
    print(f"Timeline subject counts: {summary['timeline_subject_counts']}")


if __name__ == "__main__":
    main()
