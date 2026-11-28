import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE_DIR = Path(__file__).resolve().parent.parent
LABELS = ["episodic", "semantic", "procedural", "none"]
NUM_LABELS = len(LABELS)

MODEL_NAME = "uer/chinese_roberta_L-4_H-512" 
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_all_train.jsonl"
VAL_FILE   = BASE_DIR / "data" / "train" / "memory_all_val.jsonl"
TEST_FILE  = BASE_DIR / "data" / "train" / "memory_all_test.jsonl"


INCREMENTAL_TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_incremental.jsonl"

NEW_INCREMENTAL_TRAIN_FILE = BASE_DIR / "data" / "train" / "memory_incremental_new.jsonl"

CKPT_DIR = BASE_DIR / "train" / "checkpoints" / "memory_bert"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    assert path.exists(), f"文件不存在：{path}"

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "").replace("\n", " ").strip()
            labels = obj.get("labels")

            if not text or labels is None:
                continue

            if isinstance(labels, list):
                labels = [int(x) for x in labels]
            else:
                raise ValueError("labels must be a list")

            if len(labels) != NUM_LABELS:
                raise ValueError(f"labels length: {len(labels)}, expected {NUM_LABELS}")

            obj["text"] = text
            obj["labels"] = labels
            data.append(obj)

    print(f"Loaded {path.name}: {len(data)} 条")
    return data


class MemoryDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        labels = item["labels"]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = torch.tensor(labels, dtype=torch.float)
        return enc


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = np.array(labels)
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    preds = (probs >= 0.5).astype(int)

    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
    }

def load_tokenizer_and_model(use_incremental: bool):
    if use_incremental and CKPT_DIR.exists():
        tokenizer = AutoTokenizer.from_pretrained(CKPT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(
            CKPT_DIR,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
            problem_type="multi_label_classification",
        )
    return tokenizer, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_incremental",
        action="store_true",
    )
    args = parser.parse_args()
    use_incremental=args.use_incremental
    train_data = load_jsonl(TRAIN_FILE)
    val_data   = load_jsonl(VAL_FILE)
    test_data  = load_jsonl(TEST_FILE)

    if use_incremental:
        hist_inc_data = []
        if INCREMENTAL_TRAIN_FILE.exists():
            all_hist_inc = load_jsonl(INCREMENTAL_TRAIN_FILE)
            MAX_HIST_INC_SAMPLES = 1000 

            if len(all_hist_inc) > MAX_HIST_INC_SAMPLES:
                hist_inc_data = random.sample(all_hist_inc, MAX_HIST_INC_SAMPLES)
            else:
                hist_inc_data = all_hist_inc

        else:
            new_inc_data = []
        if NEW_INCREMENTAL_TRAIN_FILE.exists():
            new_inc_data = load_jsonl(NEW_INCREMENTAL_TRAIN_FILE)
            print(f"[bert] use {len(new_inc_data)} new incremental samples for this update.")
        else:
            print("[bert] No new incremental sample files were found for this update.")

        train_data = train_data + hist_inc_data + new_inc_data
        print(f"[bert] Training set total size: {len(train_data)}")

    
    tokenizer,model = load_tokenizer_and_model(use_incremental)
    model.to(DEVICE)

    train_dataset = MemoryDataset(train_data, tokenizer, MAX_LENGTH)
    val_dataset   = MemoryDataset(val_data, tokenizer, MAX_LENGTH)
    test_dataset  = MemoryDataset(test_data, tokenizer, MAX_LENGTH)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(CKPT_DIR),     
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        greater_is_better=True,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Evaluate on the test set：")
    preds = trainer.predict(test_dataset)
    logits = preds.predictions
    labels = np.array(preds.label_ids)
    probs = 1 / (1 + np.exp(-logits))
    y_pred = (probs >= 0.5).astype(int)

    print("Micro/Macro F1:", compute_metrics((logits, labels)))
    print("Detailed classification report：")
    print(classification_report(labels, y_pred, target_names=LABELS, zero_division=0))

    print(f"Save the final model to：{CKPT_DIR}")
    trainer.save_model(CKPT_DIR)
    tokenizer.save_pretrained(CKPT_DIR)


if __name__ == "__main__":
    main()
