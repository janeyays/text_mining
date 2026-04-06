from pathlib import Path
import json, shutil

base = Path("topic_analysis/BERTopic/bertopic_models")
for sub in sorted(base.iterdir()):
    if not sub.is_dir():
        continue
    cfg = sub / "config.json"
    if not cfg.is_file():
        print(f"skip {sub.name}: no config.json")
        continue
    data = json.load(cfg.open("r", encoding="utf8"))
    if "n_gram_range" in data:
        print(f"{sub.name}: already has n_gram_range = {data['n_gram_range']}")
        continue
    val = None
    if isinstance(data.get("vectorizer"), dict):
        val = data["vectorizer"].get("ngram_range")
    if val is None:
        val = [1, 2]  # default; change to [2,2] if you want bigrams-only
    bak = sub / "config.json.bak"
    if not bak.exists():
        shutil.copy(cfg, bak)
    data["n_gram_range"] = val
    cfg.write_text(json.dumps(data, indent=2), encoding="utf8")
    print(f"patched {sub.name}: n_gram_range = {val}")