"""
Convenience launcher that runs the summarization experiment on three public
datasets (CNN/DailyMail, GovReport, PubMed) with consistent settings.

Usage:
    python run_default_benchmarks.py [--use-gpu]

The script assumes it lives next to `summarization_experiment.py` and that the
dependencies from `requirements.txt` are installed. Pass `--use-gpu` to forward
the flag to every run; otherwise all models run on CPU.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List
import sys


EXPERIMENTS = [
    {
        "dataset_name": "cnn_dailymail",
        "dataset_config": "3.0.0",
        "split": "validation",
        "text_column": "article",
        "summary_column": "highlights",
        "sample_size": 2000,
        "output_json": "results_cnn.json",
    },
    {
        "dataset_name": "ccdv/govreport-summarization",
        "dataset_config": None,
        "split": "train",
        "text_column": "report",
        "summary_column": "summary",
        "sample_size": 2000,
        "output_json": "results_gov.json",
    },
    {
        "dataset_name": "ccdv/pubmed-summarization",
        "dataset_config": None,
        "split": "train",
        "text_column": "article",
        "summary_column": "abstract",
        "sample_size": 2000,
        "output_json": "results_pubmed.json",
    },
]


def build_command(script_path: Path, use_gpu: bool, cfg: dict) -> List[str]:
    command = [
        sys.executable,
        str(script_path),
        "--dataset-name",
        cfg["dataset_name"],
        "--split",
        cfg["split"],
        "--text-column",
        cfg["text_column"],
        "--summary-column",
        cfg["summary_column"],
        "--sample-size",
        str(cfg["sample_size"]),
        "--output-json",
        cfg["output_json"],
    ]
    if cfg["dataset_config"]:
        command.extend(["--dataset-config", cfg["dataset_config"]])
    if use_gpu:
        command.append("--use-gpu")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canned HF summarization benchmarks.")
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Forward --use-gpu to every summarization_experiment.py invocation.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).with_name("summarization_experiment.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Expected summarization_experiment.py next to this script ({script_path}).")

    for cfg in EXPERIMENTS:
        command = build_command(script_path, args.use_gpu, cfg)
        print(f"\n=== Running {cfg['dataset_name']} (split={cfg['split']}) ===")
        print(" ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
