"""
Utility script to merge multiple `results_*.json` files produced by
`summarization_experiment.py` into a single JSON payload.

Usage:
    python combine_results.py --inputs results_cnn.json results_gov.json --output combined_results.json

The combined file uses the structure:
{
  "combined": [
      {"dataset": {...}, "results": [...]},
      ...
  ]
}
which is automatically recognized by `app.py` (Streamlit dashboard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combine(inputs: List[Path]) -> dict:
    combined_payload = []
    for path in inputs:
        data = load_json(path)
        combined_payload.append(
            {
                "source_file": path.name,
                "dataset": data.get("dataset", {}),
                "results": data.get("results", []),
            }
        )
    return {"combined": combined_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine multiple summarization results JSON files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="One or more JSON files produced by summarization_experiment.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="combined_results.json",
        help="Destination JSON (default: combined_results.json)",
    )
    args = parser.parse_args()

    input_paths = [Path(path) for path in args.inputs]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    combined = combine(input_paths)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"Combined results saved to {output_path}")


if __name__ == "__main__":
    main()
