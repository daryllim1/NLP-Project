## 1. Environment Setup
- Create a virtual environment (recommended).
- Install dependencies:
  ```
  pip install -r requirements.txt
  I used python 3.11.4
  ```

## 2. Run All Benchmarks Locally
Use the helper script to reproduce the three canonical HF benchmarks in one go:
```bash
python run_default_benchmarks.py --use-gpu   # drop --use-gpu if you only have CPU
```
Add `--predictions-dir outputs/preds` if you want per-example summaries saved for each dataset (they'll be grouped under `outputs/preds/<dataset>_<split>/`).
It sequentially launches:
- `cnn_dailymail` (config `3.0.0`, split `validation`)
- `ccdv/govreport-summarization` (split `train`)
- `ccdv/pubmed-summarization` (split `train`)

Each run writes its metrics to the JSON paths defined inside `run_default_benchmarks.py`.

## 3. Streamlit Front End
 Launch an interactive dashboard for visualizing metrics and trying summaries by hand:
 ```
 streamlit run app.py
 ```
 - Automatically discovers any `results_*.json` file from `summarization_experiment.py` to compare ROUGE/BERTScore across models (switch files via the dropdown).
 - If you want to view several datasets in one table, run e.g.
   ```
   python combine_results.py --inputs results_cnn.json results_gov.json results_pubmed.json --output combined_results.json
   ```
   then open `combined_results.json` in the dashboard.
 - Paste text or upload a `.txt` file, pick a model (ELMo, Bert2Bert, or BART), and generate a live summary directly in the browser.
