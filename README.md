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
- Looks for any`results_*.json` file from `summarization_experiment.py` to compare ROUGE/BERTScore across models.
- Paste text or upload a `.txt` file, pick a model (ELMo, Bert2Bert, or BART), and generate a live summary directly in the browser.
