from pathlib import Path
import json, shutil

base = Path("topic_analysis/BERTopic/bertopic_models")
for sub in sorted(base.iterdir()):
    if not sub.is_dir(): continue
    cfg = sub / "config.json"
    if not cfg.exists():
        print(f"skip {sub.name}: no config.json")
        continue
    text = cfg.read_text(encoding="utf8")
    data = json.loads(text)
    changed = False
    for k in ("umap", "hdbscan"):
        if k in data:
            data.pop(k, None)
            changed = True
    if changed:
        bak = sub / "config.json.bak"
        if not bak.exists():
            bak.write_text(text, encoding="utf8")
        cfg.write_text(json.dumps(data, indent=2), encoding="utf8")
        print(f"patched {sub.name}: removed {['umap','hdbscan']}")