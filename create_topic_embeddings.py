from pathlib import Path
import json
import numpy as np
import torch

def create_topic_embeddings_for_model(model_dir: Path):
    cfg = json.load(open(model_dir / "config.json", "r"))
    artifacts = cfg.get("model_artifacts", {})
    emb_name = artifacts.get("embeddings", "embeddings.npy")
    docs_name = artifacts.get("documents", "documents.json")
    topics_name = artifacts.get("topics", "topics.json")

    emb_path = model_dir / emb_name
    docs_path = model_dir / docs_name
    topics_path = model_dir / topics_name
    if not (emb_path.exists() and docs_path.exists() and topics_path.exists()):
        print(f"skip {model_dir.name}: missing one of {emb_name}, {docs_name}, {topics_name}")
        return

    embeddings = np.load(emb_path)
    docs = json.load(open(docs_path, "r"))
    topics = json.load(open(topics_path, "r"))

    topic_to_idxs = {}
    for d in docs:
        tid = int(d.get("topic_id", -1))
        idx = int(d.get("doc_id"))
        topic_to_idxs.setdefault(tid, []).append(idx)

    rep = topics.get("topic_representations")
    topic_keys = sorted(int(k) for k in rep.keys()) if rep else sorted(topic_to_idxs.keys())

    centroids = []
    dim = embeddings.shape[1]
    for k in topic_keys:
        idxs = topic_to_idxs.get(k, [])
        centroid = embeddings[idxs].mean(axis=0).astype(np.float32) if idxs else np.zeros(dim, dtype=np.float32)
        centroids.append(centroid)

    topic_embeddings = np.vstack(centroids)
    out_file = model_dir / "topic_embeddings.bin"
    torch.save({"topic_embeddings": torch.from_numpy(topic_embeddings)}, str(out_file))
    print(f"WROTE {out_file} shape={topic_embeddings.shape}")

if __name__ == "__main__":
    base = Path("topic_analysis/BERTopic/bertopic_models")
    for sub in sorted(base.iterdir()):
        if sub.is_dir():
            create_topic_embeddings_for_model(sub)