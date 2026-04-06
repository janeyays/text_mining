## Data Source & Extraction

Data are fetched from the arXiv API (`http://export.arxiv.org/api/query`) by querying category codes. For each returned entry we save: `id`, `title`, `summary` (abstract), `authors`, `published`, `updated`, `primary_category`, and `all_categories`. Requests are batched and rate-limited (small sleep between queries); the notebook requests up to the configured `TARGET_PER_PRIMARY_CATEGORY` per category.

## Cleaning Steps

- HTML unescaped and lowercased text.
- Whitespace normalized (multiple spaces/newlines collapsed).
- Derived fields: `title_clean`, `abstract_clean`, and combined `text`.
- Word counts: `abstract_word_count`, `title_word_count`.
- Remove duplicate rows by exact `id`.
- Filter out very short abstracts (e.g., < 20 words).
- Handle arXiv versions: extract `base_id` (paper identifier without trailing `v#`) and `version`. Optionally:
  - keep all versions for change-tracking, or
  - deduplicate to the latest version per `base_id` (the notebook includes a `df_latest` example).

## Output files

- Raw API export: `data/arxiv_raw_metadata.csv`
- Cleaned dataset: `data/arxiv_cleaned_metadata.csv`

## Setup & Run (local)

These instructions assume Windows by default — replace with the macOS/Linux variants where noted.

1. Create & activate a Python virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies

Install the BERTopic notebook dependencies and a few extras used by the app:
```bash
pip install --upgrade pip
pip install -r topic_analysis/BERTopic/requirements.txt
pip install flask transformers requests
```

PyTorch (CPU-only example):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
For GPU installs, follow the official selector: https://pytorch.org/get-started/locally/

Optional: create a single `requirements.txt` for the project after you finish installing:
```bash
pip freeze > requirements.txt
```

3. Model files (do not commit large binaries)

- Place BERTopic model folders under `topic_analysis/BERTopic/bertopic_models/`. Each subfolder should look like `bertopic_<subcat_safe_name>/` (for example `bertopic_econ_EM/`).
- If you have a BERT multilabel classifier, place its folder at the repo root named `bert_multilabel` (or update `app.py` to point to the model path).

4. Run the app

Start the Flask server:
```bash
python app.py
```
Open http://127.0.0.1:5000 in your browser.

5. Quick API test

Example curl to call topic prediction:
```bash
curl -X POST http://127.0.0.1:5000/predict_topics \
  -H "Content-Type: application/json" \
  -d '{"title":"Test title","abstract":"Short test"}'
```

6. Suggested `.gitignore` entries

Add the following to `.gitignore` to keep large files out of git:
```
topic_analysis/BERTopic/bertopic_models/
data/
*.npy
*.bin
*.safetensors
*.pt
*.pth
.venv/
__pycache__/
.ipynb_checkpoints/
.env
```


