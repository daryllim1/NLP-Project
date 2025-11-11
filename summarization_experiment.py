"""
Summarization experiment runner for comparing multiple summarization models.

This module provides a CLI for:
  * loading a dataset (local file or Hugging Face hub dataset),
  * generating summaries with different models (transformer + ELMo extractive),
  * computing standard summarization metrics (ROUGE and BERTScore F1),
  * storing aggregated results as JSON for later analysis.

The implementation is dataset-agnostic so you can plug in your own corpus when ready.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import nltk
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import torch
from datasets import Dataset, DatasetDict, load_dataset
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

ELMO_MODULE_URL = "https://tfhub.dev/google/elmo/3"
_ELMO_MODEL: Optional[object] = None
BERTSCORE_MODEL_TYPE = "roberta-large"


class ModelType(str, Enum):
    """Supported summarization backends."""

    TRANSFORMER = "transformer"
    ELMO_EXTRACTIVE = "elmo_extractive"


@dataclass
class ModelConfig:
    """Configuration for a single summarization model."""

    name: str
    model_id: str
    tokenizer_id: Optional[str] = None
    model_type: ModelType = ModelType.TRANSFORMER
    max_input_tokens: int = 1024
    max_summary_tokens: int = 128
    min_summary_tokens: int = 16
    num_beams: int = 4
    batch_size: int = 4
    num_summary_sentences: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.model_type, str):
            self.model_type = ModelType(self.model_type)


@dataclass
class ExperimentResult:
    """Holds metric outputs for a single model."""

    model_name: str
    rouge: dict
    bertscore_f1: float
    metadata: dict = field(default_factory=dict)


def resolve_device(use_gpu: bool) -> int:
    """Return device index for transformers pipeline."""
    if use_gpu and torch.cuda.is_available():
        return 0
    return -1


def load_input_dataset(
    dataset_name: Optional[str],
    dataset_path: Optional[str],
    dataset_config: Optional[str],
    split: str,
    text_column: str,
    summary_column: str,
    sample_size: Optional[int],
) -> Dataset:
    """
    Load a dataset either from the Hugging Face hub or local storage.

    The dataset must expose the columns specified via `text_column` and `summary_column`.
    """
    if dataset_name and dataset_path:
        raise ValueError("Specify either --dataset-name or --dataset-path, not both.")
    if not dataset_name and not dataset_path:
        raise ValueError("You must provide either --dataset-name or --dataset-path.")

    if dataset_name:
        dataset = load_dataset(dataset_name, name=dataset_config, split=split)
    else:
        data_path = Path(dataset_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        if data_path.suffix in {".json", ".jsonl"}:
            dataset_dict: DatasetDict = load_dataset("json", data_files=str(data_path))
        elif data_path.suffix in {".csv", ".tsv"}:
            dataset_dict = load_dataset(
                "csv", data_files=str(data_path), delimiter="," if data_path.suffix == ".csv" else "\t"
            )
        else:
            raise ValueError(
                f"Unsupported file extension '{data_path.suffix}'. "
                "Use JSON/JSONL, CSV, or TSV formats for local datasets."
            )
        dataset = dataset_dict["train"]

    missing_columns = {text_column, summary_column}.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    if sample_size is not None and sample_size < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    return dataset


def chunked(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    """Yield successive lists of size `batch_size` from `iterable`."""
    batch: List[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def generate_transformer_summaries(
    summarizer,
    inputs: List[str],
    batch_size: int,
    max_summary_tokens: int,
    min_summary_tokens: int,
    num_beams: int,
    desc: str,
) -> List[str]:
    """Run model inference and return cleaned summaries."""
    outputs: List[str] = []
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    for batch in tqdm(
        chunked(inputs, batch_size),
        total=total_batches,
        desc=desc,
    ):
        predictions = summarizer(
            batch,
            max_length=max_summary_tokens,
            min_length=min_summary_tokens,
            truncation=True,
            num_beams=num_beams,
        )
        for prediction in predictions:
            outputs.append(prediction["summary_text"].strip())
    return outputs


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return slug or "model"


def save_predictions(
    predictions_dir: Path,
    model_cfg: ModelConfig,
    inputs: List[str],
    references: List[str],
    predictions: List[str],
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    filename = predictions_dir / f"{slugify(model_cfg.name)}.jsonl"
    with filename.open("w", encoding="utf-8") as handle:
        for idx, (source, reference, prediction) in enumerate(zip(inputs, references, predictions)):
            payload = {
                "index": idx,
                "model": model_cfg.name,
                "source": source,
                "reference": reference,
                "prediction": prediction,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def ensure_punkt() -> None:
    """Make sure the NLTK punkt tokenizer data is available."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def get_elmo_model() -> object:
    """Load the TF-Hub ELMo module once."""
    global _ELMO_MODEL
    if _ELMO_MODEL is None:
        tf.get_logger().setLevel("ERROR")
        _ELMO_MODEL = hub.load(ELMO_MODULE_URL)
    return _ELMO_MODEL


def embed_sentences_elmo(sentences: List[str]) -> np.ndarray:
    """Embed a sequence of sentences with the TF-Hub ELMo module."""
    if not sentences:
        return np.zeros((0, 1024), dtype=np.float32)
    model = get_elmo_model()
    tensor = tf.constant(sentences, dtype=tf.string)
    embeddings = model.signatures["default"](tensor)["elmo"]
    return tf.reduce_mean(embeddings, axis=1).numpy()


def summarize_document_elmo(document: str, num_sentences: int) -> str:
    """Produce an extractive summary by selecting sentences closest to the ELMo document centroid."""
    ensure_punkt()
    sentences = [sent.strip() for sent in nltk.sent_tokenize(document) if sent.strip()]
    if not sentences:
        return ""

    limit = max(1, min(num_sentences, len(sentences)))

    try:
        sentence_vectors = embed_sentences_elmo(sentences)
    except Exception:
        return " ".join(sentences[:limit])

    if sentence_vectors.size == 0:
        return " ".join(sentences[:limit])

    centroid = sentence_vectors.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)

    scores = []
    for idx, vector in enumerate(sentence_vectors):
        denom = np.linalg.norm(vector) * (centroid_norm if centroid_norm != 0 else 1.0)
        if denom == 0:
            score = float("-inf")
        else:
            score = float(np.dot(vector, centroid) / denom)
        scores.append((idx, score))

    top_indices = [idx for idx, _ in sorted(scores, key=lambda item: item[1], reverse=True)[:limit]]
    top_indices.sort()
    summary_sentences = [sentences[idx] for idx in top_indices]
    return " ".join(summary_sentences)


def summarize_with_elmo(documents: List[str], num_sentences: int, desc: str) -> List[str]:
    """Generate summaries for a batch of documents using the ELMo extractive strategy."""
    summaries: List[str] = []
    for document in tqdm(documents, desc=desc, total=len(documents)):
        summaries.append(summarize_document_elmo(document, num_sentences))
    return summaries


def evaluate_model(
    model_cfg: ModelConfig,
    dataset: Dataset,
    text_column: str,
    summary_column: str,
    device: int,
    predictions_dir: Optional[Path],
) -> ExperimentResult:
    """Generate summaries for a dataset and compute evaluation metrics."""
    inputs = dataset[text_column]
    references = dataset[summary_column]

    tokenizer_id: Optional[str] = None

    if model_cfg.model_type == ModelType.TRANSFORMER:
        tokenizer_id = model_cfg.tokenizer_id or model_cfg.model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_cfg.model_id)
        if hasattr(model.config, "max_position_embeddings") and model.config.max_position_embeddings:
            tokenizer.model_max_length = model.config.max_position_embeddings
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=model_cfg.batch_size,
            truncation=True,
            max_length=model_cfg.max_summary_tokens,
        )
        predictions = generate_transformer_summaries(
            summarizer,
            inputs,
            batch_size=model_cfg.batch_size,
            max_summary_tokens=model_cfg.max_summary_tokens,
            min_summary_tokens=model_cfg.min_summary_tokens,
            num_beams=model_cfg.num_beams,
            desc=f"Summarizing with {model_cfg.name}",
        )
    elif model_cfg.model_type == ModelType.ELMO_EXTRACTIVE:
        predictions = summarize_with_elmo(
            inputs,
            num_sentences=model_cfg.num_summary_sentences,
            desc=f"Summarizing with {model_cfg.name}",
        )
    else:
        raise ValueError(f"Unsupported model type: {model_cfg.model_type}")

    rouge_metric = load_metric("rouge")
    rouge_scores = rouge_metric.compute(predictions=predictions, references=references, use_stemmer=True)

    bertscore_metric = load_metric("bertscore")
    bert_device = "cuda:0" if device >= 0 else "cpu"
    bertscore_raw = bertscore_metric.compute(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL_TYPE,
        device=bert_device,
    )
    bertscore_f1 = float(np.mean(bertscore_raw["f1"])) if bertscore_raw["f1"] else 0.0

    metadata = {
        "model_id": model_cfg.model_id,
        "tokenizer_id": tokenizer_id,
        "model_type": model_cfg.model_type.value,
        "num_examples": len(dataset),
        "max_input_tokens": model_cfg.max_input_tokens,
        "max_summary_tokens": model_cfg.max_summary_tokens,
        "num_beams": model_cfg.num_beams,
        "num_summary_sentences": model_cfg.num_summary_sentences,
    }
    if model_cfg.model_type == ModelType.ELMO_EXTRACTIVE:
        metadata["elmo_module_url"] = ELMO_MODULE_URL
    metadata["bertscore_model_type"] = BERTSCORE_MODEL_TYPE

    if predictions_dir:
        save_predictions(predictions_dir, model_cfg, inputs, references, predictions)

    return ExperimentResult(
        model_name=model_cfg.name,
        rouge=rouge_scores,
        bertscore_f1=bertscore_f1,
        metadata=metadata,
    )


def default_model_suite() -> List[ModelConfig]:
    """Return the default comparison models (ELMo extractive, BERT2BERT, and BART)."""
    return [
        ModelConfig(
            name="ELMo-Extractive",
            model_id="tfhub/google-elmo-3",
            model_type=ModelType.ELMO_EXTRACTIVE,
            batch_size=1,
            num_summary_sentences=3,
        ),
        ModelConfig(
            name="Bert2Bert-CNN",
            model_id="patrickvonplaten/bert2bert_cnn_daily_mail",
            tokenizer_id="patrickvonplaten/bert2bert_cnn_daily_mail",
            max_input_tokens=512,
            max_summary_tokens=130,
            min_summary_tokens=20,
            num_beams=4,
            batch_size=2,
        ),
        ModelConfig(
            name="BART-Large-CNN",
            model_id="facebook/bart-large-cnn",
            tokenizer_id="facebook/bart-large-cnn",
            max_input_tokens=1024,
            max_summary_tokens=140,
            min_summary_tokens=30,
            num_beams=4,
            batch_size=2,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run summarization experiments across multiple models.")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--dataset-name", type=str, help="Hugging Face dataset identifier (e.g. cnn_dailymail).")
    data_group.add_argument("--dataset-path", type=str, help="Path to a local dataset file (JSON/JSONL, CSV, TSV).")

    parser.add_argument(
        "--dataset-config",
        type=str,
        help="Optional dataset configuration name for Hugging Face datasets (e.g. 3.0.0 for cnn_dailymail).",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test).")
    parser.add_argument("--text-column", type=str, default="article", help="Column containing source documents.")
    parser.add_argument("--summary-column", type=str, default="highlights", help="Column containing reference summaries.")
    parser.add_argument("--sample-size", type=int, help="Optional sample size for quick experimentation.")

    parser.add_argument(
        "--output-json",
        type=str,
        default="summarization_results.json",
        help="Path to write aggregated results (default: summarization_results.json).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Enable GPU execution for transformer models if CUDA is available.",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        help="Optional path to a JSON file describing custom model configs.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        help="If set, write per-example predictions for each model to this directory as JSONL files.",
    )

    return parser.parse_args()


def load_models_from_file(config_path: str) -> List[ModelConfig]:
    """Load a list of ModelConfig objects from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("Model config JSON must be a list of objects.")

    models: List[ModelConfig] = []
    for element in payload:
        models.append(ModelConfig(**element))
    return models


def main() -> None:
    args = parse_args()
    device = resolve_device(args.use_gpu)

    dataset = load_input_dataset(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        split=args.split,
        text_column=args.text_column,
        summary_column=args.summary_column,
        sample_size=args.sample_size,
    )

    models = load_models_from_file(args.models_config) if args.models_config else default_model_suite()
    predictions_dir = Path(args.predictions_dir) if args.predictions_dir else None

    results: List[ExperimentResult] = []
    for model_cfg in models:
        result = evaluate_model(
            model_cfg,
            dataset,
            args.text_column,
            args.summary_column,
            device,
            predictions_dir,
        )
        results.append(result)

    serializable_results = {
        "dataset": {
            "source": args.dataset_name or args.dataset_path,
            "split": args.split,
            "text_column": args.text_column,
            "summary_column": args.summary_column,
            "num_examples": len(dataset),
        },
        "results": [asdict(result) for result in results],
    }

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(serializable_results, indent=2), encoding="utf-8")
    print(f"Saved experiment results to {output_path.resolve()}")


if __name__ == "__main__":
    main()
