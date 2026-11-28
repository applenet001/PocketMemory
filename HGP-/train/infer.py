import json
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent

LABELS = ["episodic", "semantic", "procedural",  "none"]

CKPT_DIR = BASE_DIR / "train" / "checkpoints" / "memory_bert"

ORIG_TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_all_train.jsonl"

INCREMENTAL_TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_incremental.jsonl"
NEW_INCREMENTAL_TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_incremental_new.jsonl"

EMB_MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
SIM_THRESHOLD = 0.90 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 128

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
        _model.to(DEVICE)
        _model.eval()
    return _tokenizer, _model


def predict_text(text: str, thr_for_label: float = 0.5) -> Tuple[List[str], np.ndarray]:
    tokenizer, model = _load_model()

    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits[0]  # (num_labels,)

    probs = torch.sigmoid(logits).detach().cpu().numpy()

    pred = (probs >= thr_for_label).astype(int)
    labels = [LABELS[i] for i, v in enumerate(pred) if v == 1]

    return labels, probs

def _load_texts_from_jsonl(path: Path) -> List[str]:
    texts: List[str] = []
    if not path.exists():
        return texts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            t = str(obj.get("text", "")).strip()
            if t:
                texts.append(t)
    return texts


def semantic_dedup_new_samples(new_samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not new_samples:
        return []

    base_texts: List[str] = []
    base_texts += _load_texts_from_jsonl(ORIG_TRAIN_FILE)
    base_texts += _load_texts_from_jsonl(INCREMENTAL_TRAIN_FILE)

    new_texts = [s["text"] for s in new_samples]

    model = SentenceTransformer(EMB_MODEL_NAME)

    if base_texts:
        base_embs = model.encode(
            base_texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    else:
        base_embs = None

    new_embs = model.encode(
        new_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    kept_samples: List[Dict[str, Any]] = []
    kept_embs: List[np.ndarray] = []

    for i, emb in enumerate(new_embs):
        dup = False

        if base_embs is not None and base_embs.shape[0] > 0:
            sims_base = base_embs @ emb  
            if float(sims_base.max()) >= SIM_THRESHOLD:
                dup = True

        if dup:
            continue

        if kept_embs:
            kept_matrix = np.vstack(kept_embs)
            sims_new = kept_matrix @ emb
            if float(sims_new.max()) >= SIM_THRESHOLD:
                continue

        kept_samples.append(new_samples[i])
        kept_embs.append(emb)

    return kept_samples

def incremental_update(new_data_path: Path, out_path: Optional[Path] = None) -> None:
    if out_path is None:
        out_path = INCREMENTAL_TRAIN_FILE

    if not new_data_path.exists():
        return

    raw_new_samples: List[Dict[str, Any]] = []
    with new_data_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            text = str(obj.get("text", "")).strip()
            labels = obj.get("labels")

            if not text or labels is None:
                continue

            if isinstance(labels, list):
                try:
                    labels = [int(x) for x in labels]
                except Exception:
                    continue
            else:
                continue

            if len(labels) != len(LABELS):
                continue

            raw_new_samples.append({"text": text, "labels": labels})

    if not raw_new_samples:
        return

    new_samples = semantic_dedup_new_samples(raw_new_samples)
    if not new_samples:
        return

    NEW_INCREMENTAL_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NEW_INCREMENTAL_TRAIN_FILE.open("w", encoding="utf-8") as f_new:
        for obj in new_samples:
            f_new.write(json.dumps(obj, ensure_ascii=False) + "\n")


    out_path.parent.mkdir(parents=True, exist_ok=True)

    before_cnt = 0
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            before_cnt = sum(1 for _ in f)

    bert_script = Path(__file__).parent / "bert.py"
    if not bert_script.exists():
        return

    try:
        subprocess.run(
            [sys.executable, str(bert_script), "--use_incremental"],
            cwd=str(bert_script.parent),
            check=True,
        )
        global _tokenizer, _model
        _tokenizer = None
        _model = None

        from train.adaptive_threshold import record_training
        record_training()

        with out_path.open("a", encoding="utf-8") as fout:
            for obj in new_samples:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except subprocess.CalledProcessError:
        pass
