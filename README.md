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