"""
Streamlit front end for summarization experiments.

Features:
1. Upload an aggregated results JSON (output of summarization_experiment.py) and
   view model metrics in a table.
2. Paste text or upload a .txt file, choose one of the default models, and
   generate a fresh summary on the fly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from summarization_experiment import (  # type: ignore
    ModelConfig,
    ModelType,
    default_model_suite,
    ensure_punkt,
    summarize_document_elmo,
)


st.set_page_config(page_title="Summarization Benchmark Dashboard", layout="wide")


@st.cache_resource(show_spinner=True)
def load_transformer_pipeline(model_id: str, tokenizer_id: Optional[str], device_idx: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id or model_id, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=device_idx)


def summarize_text(model_cfg: ModelConfig, text: str, device_idx: int) -> str:
    if model_cfg.model_type == ModelType.ELMO_EXTRACTIVE:
        ensure_punkt()
        return summarize_document_elmo(text, model_cfg.num_summary_sentences)

    summarizer = load_transformer_pipeline(model_cfg.model_id, model_cfg.tokenizer_id, device_idx)
    result = summarizer(
        text,
        max_length=model_cfg.max_summary_tokens,
        min_length=model_cfg.min_summary_tokens,
        truncation=True,
        num_beams=model_cfg.num_beams,
    )
    return result[0]["summary_text"].strip()


def discover_result_files() -> List[Path]:
    return sorted(Path(".").glob("results_*.json"))


def render_results_table(results_payload: dict) -> None:
    rows = []
    for item in results_payload.get("results", []):
        rouge = item.get("rouge", {})
        rows.append(
            {
                "Model": item.get("model_name"),
                "ROUGE-1": round(rouge.get("rouge1", 0.0), 4),
                "ROUGE-2": round(rouge.get("rouge2", 0.0), 4),
                "ROUGE-L": round(rouge.get("rougeL", 0.0), 4),
                "BERTScore F1": round(item.get("bertscore_f1", 0.0), 4),
            }
        )
    if not rows:
        st.info("No results found in the uploaded JSON.")
        return
    df = pd.DataFrame(rows)
    st.subheader("Model Comparison")
    st.table(df)


def read_uploaded_json(uploaded_file) -> Optional[dict]:
    if uploaded_file is None:
        return None
    try:
        return json.load(uploaded_file)
    except json.JSONDecodeError:
        st.error("Uploaded file is not valid JSON.")
        return None


def get_text_input(uploaded_txt) -> Optional[str]:
    manual_text = st.text_area("Or paste raw text here", height=250)
    if uploaded_txt is not None:
        try:
            return uploaded_txt.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Failed to read uploaded file: {exc}")
            return manual_text
    return manual_text or None


def render_results_tab() -> None:
    st.subheader("Model Comparison")
    local_files = discover_result_files()
    col1, col2 = st.columns([2, 1])

    selected_path: Optional[Path] = None
    if local_files:
        options = [p.name for p in local_files]
        default_index = 0
        selected_name = col1.selectbox("Local results files", options, index=default_index)
        selected_path = next((p for p in local_files if p.name == selected_name), None)
    else:
        col1.info("No local results_*.json files detected.")

    uploaded_results = col2.file_uploader("...or upload a JSON", type=["json"], accept_multiple_files=False)

    payload = None
    if uploaded_results is not None:
        payload = read_uploaded_json(uploaded_results)
    elif selected_path is not None:
        payload = json.loads(selected_path.read_text(encoding="utf-8"))

    if payload:
        render_results_table(payload)
    else:
        st.info(
            "Drop a `results_*.json` file from `summarization_experiment.py` or place one "
            "next to this app (filenames like `results_cnn.json` are detected automatically)."
        )


def render_summarizer_tab() -> None:
    st.subheader("Interactive Summarization")
    models: List[ModelConfig] = default_model_suite()
    model_names = [model.name for model in models]
    selected_model_name = st.selectbox("Choose a model", model_names)
    selected_model = next(model for model in models if model.name == selected_model_name)

    uploaded_txt = st.file_uploader("Optional: upload a .txt file", type=["txt"], accept_multiple_files=False)
    text_input = get_text_input(uploaded_txt)

    device_idx = 0 if torch.cuda.is_available() else -1

    summarize_click = st.button("Summarize")
    if summarize_click and text_input:
        with st.spinner(f"Generating summary with {selected_model.name}..."):
            summary = summarize_text(selected_model, text_input, device_idx)
        st.subheader("Summary")
        st.write(summary)
    elif summarize_click:
        st.warning("Please supply text via the textarea or upload a .txt file.")


def main() -> None:
    st.title("Summarization Benchmark + Playground")
    st.write(
        "Drop in your evaluation results and experiment with the bundled models—all from one screen."
    )

    results_tab, summarizer_tab = st.tabs(["📊 Results Dashboard", "📝 Summarizer"])
    with results_tab:
        render_results_tab()
    with summarizer_tab:
        render_summarizer_tab()


if __name__ == "__main__":
    main()
