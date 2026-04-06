"""
bertopic_inference.py
---------------------
Predict topics for a research paper (title + abstract) using the
pre-trained BERTopic models saved by the notebook.

Quick start
-----------
    from bertopic_inference import BERTopicPredictor

    predictor = BERTopicPredictor(model_dir="bertopic_models/")

    # Query a specific subcategory:
    result = predictor.predict(
        title="...",
        abstract="...",
        subcategory="eess.SP",
    )
    print(result)

    # Or query ALL loaded subcategories at once:
    results = predictor.predict(title="...", abstract="...")
    for sc, pred in results.items():
        print(sc, pred)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────────────────────
# Acronym expansion  (mirrors the notebook's expand_acronyms_keep_both_safe)
# ─────────────────────────────────────────────────────────────────────────────

def _norm(token: str) -> str:
    """Lowercase and keep only alphanumerics + hyphens."""
    return re.sub(r"[^a-z0-9\-]", "", token.lower())


def _valid_acronym(acr: str) -> bool:
    return acr.isalpha() and 2 <= len(acr) <= 10


def _initials(words: list) -> str:
    return "".join(w[0] for w in map(_norm, words) if w)


def _best_suffix(full_term: str, acronym: str) -> Optional[str]:
    """Find the shortest suffix of `full_term` whose initials match `acronym`."""
    words = [w for w in full_term.split() if _norm(w)]
    acr_lower = acronym.lower()
    for start in range(len(words)):
        sfx = words[start:]
        if len(sfx) >= 2 and _initials(sfx) == acr_lower:
            return " ".join(sfx)
    return None


def expand_acronyms(text: str) -> str:
    """
    Expand 'term (ACR)' patterns into 'acr term_underscored' and
    replace subsequent bare occurrences of ACR consistently.
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    pattern = re.compile(
        r"((?:\b[a-z][a-z0-9\-]*\b(?:\s+|$)){1,8})\(([a-z0-9\-]{2,10})\)"
    )
    amap: dict = {}
    for m in pattern.finditer(t):
        ft, ac = m.group(1).strip(), m.group(2).strip()
        if _valid_acronym(ac):
            bd = _best_suffix(ft, ac)
            if bd:
                amap[ac] = "_".join(bd.split())

    def _replace(m: re.Match) -> str:
        ac = m.group(2).strip()
        return f"{ac} {amap[ac]}" if ac in amap else m.group(0)

    t = pattern.sub(_replace, t)
    for ac, fc in amap.items():
        t = re.sub(
            rf"\b{re.escape(ac)}\b(?!\s+{re.escape(fc)})",
            f"{ac} {fc}", t,
        )
    return t


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TopicPrediction:
    """Holds the predicted topic for one (subcategory, paper) pair."""
    subcategory:  str
    topic_id:     int
    topic_label:  str        # BERTopic auto-generated name, e.g. '2_ecg_signal_heart'
    top_keywords: list       # top-N c-TF-IDF keywords for this topic
    probability:  float      # confidence in the assignment
    is_outlier:   bool = False          # True when topic_id == -1
    all_probs:    dict = field(default_factory=dict)  # optional full prob vector

    def __str__(self) -> str:
        status = "OUTLIER – no strong topic match" if self.is_outlier else "Matched"
        lines = [
            f"  Status      : {status}",
            f"  Subcategory : {self.subcategory}",
            f"  Topic ID    : {self.topic_id}",
            f"  Label       : {self.topic_label}",
            f"  Keywords    : {', '.join(self.top_keywords)}",
            f"  Probability : {self.probability:.3f}",
        ]
        if self.all_probs:
            lines.append("  Topic probability breakdown:")
            for tid, prob in sorted(self.all_probs.items(), key=lambda x: -x[1]):
                lines.append(f"    Topic {tid:>3}: {prob:.4f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main predictor class
# ─────────────────────────────────────────────────────────────────────────────

class BERTopicPredictor:
    """
    Load saved BERTopic models and predict topics for new research papers.

    Parameters
    ----------
    model_dir : str
        Directory containing the BERTopic sub-folders saved by the
        notebook, e.g. bertopic_econ_TH/, bertopic_eess_SY/ ...
    embedding_model_name : str
        Must match the SentenceTransformer used during training
        (default: all-MiniLM-L6-v2).
    top_n_keywords : int
        How many top c-TF-IDF keywords to surface per predicted topic.
    """

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_dir: str = "bertopic_models/",
        embedding_model_name: str = EMBEDDING_MODEL,
        top_n_keywords: int = 10,
    ):
        self.model_dir       = model_dir
        self.top_n_keywords  = top_n_keywords
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.models: dict    = {}
        self._load_models()

    # ── model loading ──────────────────────────────────────────────────────────

    def _load_models(self) -> None:
        """
        Walk model_dir and load every BERTopic sub-folder.
        Folder name format expected:  bertopic_<subcat_safe_name>
        e.g. bertopic_econ_TH  ->  subcategory key  econ.TH
        """
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"model_dir not found: {self.model_dir}")

        for entry in sorted(os.listdir(self.model_dir)):
            full_path = os.path.join(self.model_dir, entry)
            if not os.path.isdir(full_path):
                continue
            # Reverse the safe-name conversion used in the notebook:
            #   bertopic_econ_TH  ->  econ.TH
            subcat = entry.removeprefix("bertopic_").replace("_", ".", 1)
            try:
                self.models[subcat] = BERTopic.load(
                    full_path, embedding_model=self.embedding_model
                )
                print(f"  Loaded: {subcat}")
            except Exception as exc:
                print(f"  WARNING – could not load {entry}: {exc}")

        if not self.models:
            raise RuntimeError(
                "No BERTopic models were loaded. "
                "Check that model_dir contains bertopic_* sub-folders."
            )

    # ── text preprocessing ─────────────────────────────────────────────────────

    def _preprocess(self, title: str, abstract: str) -> str:
        """Concatenate title + abstract and run acronym expansion."""
        combined = f"{title.strip()} {abstract.strip()}"
        return expand_acronyms(combined)

    def _embed(self, text: str) -> np.ndarray:
        """Produce a (1, 384) sentence embedding."""
        return self.embedding_model.encode([text])

    # ── public API ─────────────────────────────────────────────────────────────

    def predict(
        self,
        title: str,
        abstract: str,
        subcategory: Optional[str] = None,
        return_all_probs: bool = False,
    ):
        """
        Predict the BERTopic topic(s) for a single research paper.

        Parameters
        ----------
        title : str
            Paper title.
        abstract : str
            Paper abstract.
        subcategory : str or None
            If provided (e.g. 'econ.TH'), only that domain model is
            used and a single TopicPrediction is returned.
            If None, every loaded model is queried and a
            dict[str, TopicPrediction] is returned.
        return_all_probs : bool
            When True, include the full per-topic probability vector
            in the result (TopicPrediction.all_probs).

        Returns
        -------
        TopicPrediction                  – if subcategory is specified
        dict[str, TopicPrediction]       – if subcategory is None
        """
        text      = self._preprocess(title, abstract)
        embedding = self._embed(text)

        if subcategory is not None:
            if subcategory not in self.models:
                raise KeyError(
                    f"Unknown subcategory '{subcategory}'. "
                    f"Available: {list(self.models)}"
                )
            return self._predict_one(
                subcategory, self.models[subcategory], embedding, return_all_probs
            )

        return {
            sc: self._predict_one(sc, model, embedding, return_all_probs)
            for sc, model in self.models.items()
        }

    def _predict_one(
        self,
        subcategory: str,
        model: BERTopic,
        embedding: np.ndarray,
        return_all_probs: bool,
    ) -> TopicPrediction:
        # BERTopic.transform expects a list of strings even if we only
        # care about the embedding; 'placeholder' is ignored internally.
        topics, probs = model.transform(["placeholder"], embeddings=embedding)
        topic_id = int(topics[0])

        # probs[0] may be a scalar or a 1-D probability array
        prob_val   = probs[0]
        prob_array = (np.atleast_1d(prob_val)
                      if hasattr(prob_val, "__len__")
                      else np.array([prob_val]))
        best_prob  = float(prob_array.max()) if prob_array.size > 1 else float(prob_array[0])

        # Top keywords for the assigned topic
        raw_topic = model.get_topic(topic_id) if topic_id != -1 else []
        keywords  = [w for w, _ in raw_topic[: self.top_n_keywords]] if raw_topic else []

        # Human-readable label
        topic_info = model.get_topic_info()
        label_row  = topic_info[topic_info["Topic"] == topic_id]
        label      = label_row["Name"].values[0] if len(label_row) else str(topic_id)

        # Optional full probability breakdown
        all_probs: dict = {}
        if return_all_probs and prob_array.size > 1:
            all_probs = {i: float(p) for i, p in enumerate(prob_array)}

        return TopicPrediction(
            subcategory=subcategory,
            topic_id=topic_id,
            topic_label=label,
            top_keywords=keywords,
            probability=best_prob,
            is_outlier=(topic_id == -1),
            all_probs=all_probs,
        )

    @property
    def available_subcategories(self) -> list:
        """List of subcategory keys for loaded models."""
        return list(self.models.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Demo  (python bertopic_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Adjust to wherever your notebook wrote the models:
    #   /content/drive/MyDrive/textmining_proj/bertopic_models/
    MODEL_DIR = "bertopic_models/"

    predictor = BERTopicPredictor(model_dir=MODEL_DIR)
    print(f"\nLoaded subcategories: {predictor.available_subcategories}\n")

    title = (
        "Convolutional Neural Networks for Real-Time Cardiac Arrhythmia "
        "Classification from ECG Signals"
    )
    abstract = (
        "We present a deep learning (DL) approach that leverages a "
        "convolutional neural network (CNN) to classify cardiac arrhythmias "
        "directly from raw electrocardiogram (ECG) waveforms. "
        "The model is trained on the PhysioNet dataset and achieves 97.3% "
        "accuracy across five arrhythmia classes, outperforming classical "
        "support vector machine (SVM) baselines by 4.1 percentage points. "
        "Real-time inference on an embedded ARM Cortex-M7 runs under 12 ms per beat."
    )

    # ── single subcategory ───────────────────────────────────────────────────
    print("=" * 55)
    print("Single-subcategory prediction  (eess.SP)")
    print("=" * 55)
    single = predictor.predict(
        title, abstract, subcategory="eess.SP", return_all_probs=True
    )
    print(single)

    # ── all subcategories ────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("All-subcategory prediction")
    print("=" * 55)
    all_results = predictor.predict(title, abstract)
    for sc, pred in all_results.items():
        print(f"\n[{sc}]")
        print(pred)
