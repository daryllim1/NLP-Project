"""
Utility script to mirror the Colab dataset peek locally.

Usage:
    python load_hf_samples.py

It downloads three benchmark summarization datasets (CNN/DailyMail, GovReport,
and PubMed) and prints the first training example from each so you can verify
the column names (`article`/`highlights`, `report`/`summary`, etc.).
"""

from __future__ import annotations

from datasets import load_dataset


def preview_dataset(name: str, config: str | None, split: str, text_key: str, summary_key: str) -> None:
    dataset = load_dataset(name, name=config, split=split)
    sample = dataset[0]
    text_snippet = sample[text_key][:500]
    summary_snippet = sample[summary_key][:400]

    print(f"=== {name} ({split}) ===")
    print("Keys:", sample.keys())
    print(text_snippet)
    print("—" * 10)
    print(summary_snippet)
    print()


def main() -> None:
    preview_dataset("cnn_dailymail", "3.0.0", "train", "article", "highlights")
    preview_dataset("ccdv/govreport-summarization", None, "train", "report", "summary")
    preview_dataset("ccdv/pubmed-summarization", None, "train", "article", "abstract")


if __name__ == "__main__":
    main()
