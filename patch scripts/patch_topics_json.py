from pathlib import Path
import json, shutil

base = Path("topic_analysis/BERTopic/bertopic_models")
for sub in sorted(base.iterdir()):
    if not sub.is_dir(): 
        continue
    tfile = sub / "topics.json"
    if not tfile.exists():
        print(f"skip {sub.name}: no topics.json")
        continue

    orig_text = tfile.read_text(encoding="utf8")
    data = json.loads(orig_text)

    # already in correct format?
    if isinstance(data, dict) and "topic_representations" in data:
        print(f"already ok: {sub.name}")
        continue

    topic_reprs = {}
    topic_sizes = {}
    topic_labels = {}
    topics_list = []
    topic_mapper = []

    # handle dict keyed by topic-id
    if isinstance(data, dict):
        for key, val in data.items():
            try:
                tid = int(key)
            except Exception:
                continue
            topics_list.append(tid)

            # extract keywords (support a few possible shapes)
            kw_list = []
            if isinstance(val, dict):
                kws = val.get("keywords") or val.get("topic_keywords") or val.get("topic_representation")
                if isinstance(kws, list):
                    for item in kws:
                        if isinstance(item, dict):
                            w = item.get("word") or item.get("keyword")
                            s = item.get("score") or item.get("weight") or 0.0
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            w, s = item[0], item[1]
                        else:
                            continue
                        if w is None:
                            continue
                        kw_list.append([str(w), float(s)])
                topic_sizes[str(tid)] = int(val.get("count", val.get("size", 0)))
                topic_labels[str(tid)] = val.get("name", str(tid))
            topic_reprs[str(tid)] = kw_list

    # handle list-of-objects (less likely)
    elif isinstance(data, list):
        for obj in data:
            if not isinstance(obj, dict):
                continue
            tid = int(obj.get("topic_id", obj.get("topic", -1)))
            topics_list.append(tid)
            kws = obj.get("keywords") or obj.get("topic_keywords")
            kw_list = []
            if isinstance(kws, list):
                for item in kws:
                    if isinstance(item, dict):
                        w = item.get("word") or item.get("keyword")
                        s = item.get("score") or item.get("weight") or 0.0
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        w, s = item[0], item[1]
                    else:
                        continue
                    if w is None:
                        continue
                    kw_list.append([str(w), float(s)])
            topic_reprs[str(tid)] = kw_list
            topic_sizes[str(tid)] = int(obj.get("count", obj.get("size", 0)))
            topic_labels[str(tid)] = obj.get("name", str(tid))

    topics_list = sorted(set(topics_list))
    for tid in topics_list:
        if str(tid) not in topic_reprs:
            topic_reprs[str(tid)] = []
        if str(tid) not in topic_sizes:
            topic_sizes[str(tid)] = 0
        if str(tid) not in topic_labels:
            topic_labels[str(tid)] = str(tid)
        topic_mapper.append([tid, tid])

    new = {
        "topic_representations": topic_reprs,
        "topics": topics_list,
        "topic_sizes": topic_sizes,
        "topic_mapper": topic_mapper,
        "topic_labels": topic_labels,
        "custom_labels": None,
        "_outliers": 1 if "-1" in topic_sizes else 0,
        "topic_aspects": {},
    }

    bak = sub / "topics.json.bak"
    if not bak.exists():
        bak.write_text(orig_text, encoding="utf8")
    tfile.write_text(json.dumps(new, indent=2), encoding="utf8")
    print(f"patched {sub.name}")