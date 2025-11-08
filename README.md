# Summarization Model Benchmark

Compare multiple summarizers (extractive + abstractive) on a shared dataset with ROUGE and BERTScore F1. The project is data-agnostic so you can plug in any corpus once it is ready.

## 1. Environment Setup
- Create a virtual environment (recommended).
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## 2. Dataset Expectations
- Provide either a Hugging Face dataset name (`--dataset-name`) or a local file (`--dataset-path`).
- When a dataset offers multiple configurations, pass the variant with `--dataset-config` (e.g., `--dataset-config 3.0.0` for `cnn_dailymail`).
- Local files should be JSON/JSONL, CSV, or TSV with at least two columns:
  - source document column (default `article`)
  - reference summary column (default `highlights`)
- Use `--text-column` and `--summary-column` to override defaults.
- Optional `--sample-size` limits the number of examples for quick dry runs.

## 3. Run the Experiment
Example using a Hugging Face dataset:
```bash
python summarization_experiment.py \
  --dataset-name cnn_dailymail \
  --split test \
  --text-column article \
  --summary-column highlights \
  --sample-size 32 \
  --use-gpu \
  --predictions-dir outputs/predictions
```

Example using a local JSONL file:
```bash
python summarization_experiment.py \
  --dataset-path data/my_dataset.jsonl \
  --text-column source \
  --summary-column reference
```

Results are written to `summarization_results.json` by default. Use `--output-json` to customize.
- Each entry in the output JSON reports ROUGE (1/2/L/Lsum) and BERTScore F1 (computed with `roberta-large`).
- Pass `--predictions-dir some_folder` to capture the per-example source/reference/prediction triples for each model as JSONL files (handy for manual review).

## 4. Custom Model Suites
- The default run compares:
  1. `ELMo-Extractive` – sentence ranking with TensorFlow Hub's ELMo encoder
  2. `patrickvonplaten/bert2bert_cnn_daily_mail`
  3. `facebook/bart-large-cnn`
- Supply your own JSON file via `--models-config` to tweak or extend the model list. Expected schema:
  ```json
  [
    {
      "name": "Custom Model",
      "model_id": "org/model-name",
      "tokenizer_id": "org/tokenizer-name",
      "model_type": "transformer",
      "max_input_tokens": 512,
      "max_summary_tokens": 128,
      "min_summary_tokens": 16,
      "num_beams": 4,
      "batch_size": 2
    }
  ]
  ```
- To add another extractive model using the bundled strategy, set `"model_type": "elmo_extractive"` and optionally `"num_summary_sentences": 3`.
- The script automatically downloads the TF Hub ELMo weights, the NLTK `punkt` tokenizers, and the `roberta-large` checkpoint used for BERTScore on first run.

## 5. Next Ideas
- Plug in additional metrics (e.g., QA-based faithfulness checks) by extending `evaluate_model`.
- Persist per-example outputs for qualitative error analysis.
- Integrate experiment tracking (Weights & Biases, MLflow) for repeated runs.

## 6. Run All Benchmarks Locally
Use the helper script to reproduce the three canonical HF benchmarks in one go:
```bash
python run_default_benchmarks.py --use-gpu   # drop --use-gpu if you only have CPU
```
It sequentially launches:
- `cnn_dailymail` (config `3.0.0`, split `validation`)
- `ccdv/govreport-summarization` (split `train`)
- `ccdv/pubmed-summarization` (split `train`)

Each run writes its metrics to the JSON paths defined inside `run_default_benchmarks.py`.

## 7. Streamlit Front End
Launch an interactive dashboard for visualizing metrics and trying summaries by hand:
```bash
streamlit run app.py
```
- Upload any `results_*.json` file from `summarization_experiment.py` to compare ROUGE/BERTScore across models.
- Paste text or upload a `.txt` file, pick a model (ELMo, Bert2Bert, or BART), and generate a live summary directly in the browser.
