from pathlib import Path
import json, shutil

base = Path("topic_analysis/BERTopic/bertopic_models")
for sub in sorted(base.iterdir()):
    if not sub.is_dir():
        continue
    cfg = sub / "config.json"
    if not cfg.exists():
        print(f"skip {sub.name}: no config.json")
        continue

    original = cfg.read_text(encoding="utf8")
    data = json.loads(original)
    changed = False

    # 1) map min_cluster_size -> min_topic_size
    if "min_cluster_size" in data:
        data["min_topic_size"] = data.pop("min_cluster_size")
        changed = True

    # 2) lift nested 'bertopic' keys to top-level (common small params)
    if isinstance(data.get("bertopic"), dict):
        for k, v in data["bertopic"].items():
            if k not in data:
                data[k] = v
                changed = True
        data.pop("bertopic", None)

    # 3) remove problematic object-typed keys that become unexpected kwargs
    for k in ("vectorizer", "umap", "hdbscan", "model_artifacts", "evaluation", "subcategory"):
        if k in data:
            data.pop(k, None)
            changed = True

    # 4) ensure n_gram_range exists (safe default)
    if "n_gram_range" not in data:
        # try to reuse vectorizer.ngram_range if it existed previously
        data["n_gram_range"] = [1, 2]
        changed = True

    if changed:
        bak = sub / "config.json.bak"
        if not bak.exists():
            bak.write_text(original, encoding="utf8")
        cfg.write_text(json.dumps(data, indent=2), encoding="utf8")
        print(f"patched {sub.name}")