import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from llm_api import get_llm_client
from train.infer import predict_text, incremental_update, LABELS


BASE_DIR = Path(__file__).resolve().parent.parent

BUFFER_PATH = BASE_DIR / "data" / "new_labeled_buffer.jsonl"

BUFFER_LIMIT = 100
llm_client = get_llm_client()


def call_llm_for_classification(text: str) -> List[str]:

    system_prompt = "You are an assistant responsible for classifying text into memory types; you only output JSON."

    prompt = f"""
There are four types of memory (labeled in English):
1) episodic (episodic memory)
   Describes events or actions involving a specific subject (such as “user,” “I,” “system,” “agent,” etc.) in a particular context.
   May explicitly or implicitly include elements such as “when, where, by whom, to whom, and what happened.”
   Common forms resemble “logs”: a person or system performed a specific action on this occasion, or expressed a certain state, feeling, or opinion.

2) Semantic memory
   Does not refer to a specific experience, but rather stable units of knowledge that can be reused across time and tasks.
   Used to state general facts, definitions, concepts, attributes, universal laws, or the user’s long-term preferences and habits.
   This type of knowledge is no longer tied to specific one-time events; it can stand independently of the context of “who did what on this occasion.”

3) Procedural Memory
   Describes skills and operational procedures that answer “how to do it,” emphasizing the mapping from conditions to actions, as well as action sequences, rules, and strategies,
   rather than single events or abstract facts.
   This content can be directly reused by agents as “operational guidelines/strategic rules.”
   
4) none (Fragmented Memory)
   Content that is too brief, severely lacking in context, or semantically unclear, making it difficult to reliably classify into the above three categories.
   Examples: exclamatory sentences, garbled text, or severely truncated fragments whose meaning cannot be determined.
   Use “none” when the content truly cannot be reasonably classified as episodic, semantic, or procedural.

Classification Rules:
You must output a JSON object containing the “labels” field, whose value is an array of strings.
  For example: {{“labels”: [“episodic”, “semantic”]}} or {{‘labels’: [“none”]}}.
A single text segment may correspond to multiple labels.
Priority Rules (Very Important):
  If the text describes a specific action or event, it must include the “episodic” label.
  If the text primarily explains general facts, concepts, or reveals user preferences and habits, it can be labeled as “semantic”.
  If the text describes directly reusable steps, processes, or “if-then” rules, it can be labeled as “procedural”.
  If the text describes the current FSM state or control variables (e.g., current phase, whether the current subtask is complete, current retry count, etc.), it can be labeled as “working”.

Please classify the following text and output only JSON (no additional explanations):
{text}

The only allowed labels are: episodic, semantic, procedural, none
""".strip()

    
    try:
        result = llm_client.call_llm_with_json(prompt, temperature=0.2, system_prompt=system_prompt)
        labels = result.get("labels", [])
    except Exception:
        labels = ["none"]

    labels = [lab for lab in labels if lab in LABELS]
    if not labels:
        labels = ["none"]

    return labels


class InferencePipeline:
    def __init__(
        self,
        threshold: Optional[float] = None,
        buffer_path: Path = BUFFER_PATH,
        buffer_limit: int = BUFFER_LIMIT,
    ):
        if threshold is None:
            from train.adaptive_threshold import get_current_classifier_threshold
            self.threshold = get_current_classifier_threshold()
            self.use_dynamic_threshold = True
        else:
            self.threshold = threshold
            self.use_dynamic_threshold = False
        
        self.buffer_path = buffer_path
        self.buffer_limit = buffer_limit
    
    def get_current_threshold(self) -> float:
        if self.use_dynamic_threshold:
            from train.adaptive_threshold import get_current_classifier_threshold
            self.threshold = get_current_classifier_threshold()
        return self.threshold

    def infer(self, text: str) -> Dict[str, Any]:
        labels, scores = predict_text(text)  # scores: np.array
        scores_arr = np.asarray(scores, dtype=float).reshape(-1)
        if scores_arr.size == 0:
            scores_arr = np.zeros(len(LABELS), dtype=float)

        max_confidence = float(scores_arr.max()) if scores_arr.size > 0 else 0.0

        result: Dict[str, Any] = {
            "text": text,
            "labels": labels,
            "scores": scores_arr.tolist(),
            "source": "classifier",
            "used_llm": False,
            "confidence": max_confidence,
        }
        if (not labels) or max_confidence < self.threshold:
            llm_labels = call_llm_for_classification(text)
            result["labels"] = llm_labels
            result["source"] = "llm"
            result["used_llm"] = True

        if result["used_llm"]:
            self._log_and_maybe_buffer(result)

        confidence_scores = {
            LABELS[i]: float(scores_arr[i])
            for i in range(len(LABELS))
        }

        return result

    def _labels_to_vector(self, labels: List[str]) -> List[int]:
        vec = [0] * len(LABELS)
        if not labels:
            vec[-1] = 1
            return vec

        for lab in labels:
            if lab in LABELS:
                idx = LABELS.index(lab)
                vec[idx] = 1

        if sum(vec) == 0:
            vec[-1] = 1

        return vec

    def _log_and_maybe_buffer(self, result: Dict[str, Any]) -> None:
        label_vector = self._labels_to_vector(result["labels"])
        record = {
            "text": result["text"],
            "labels": label_vector,
            "raw_labels": result["labels"],
            "source": result["source"],
            "confidence": result["confidence"],
            "scores": result["scores"],
        }

        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        with self.buffer_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        try:
            with self.buffer_path.open("r", encoding="utf-8") as f:
                line_count = sum(1 for _ in f)
        except FileNotFoundError:
            line_count = 0

        if line_count >= self.buffer_limit:
            incremental_update(self.buffer_path)
        
            
            self.buffer_path.unlink(missing_ok=True)


