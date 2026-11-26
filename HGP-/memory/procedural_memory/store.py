import json
import uuid
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
@dataclass
class ProcedureRule:
    id: str
    title: str
    if_conditions: List[str]
    then_steps: List[str]
    tree: Dict[str, Any]
    tags: List[str]
    source_text: str
    meta: Dict[str, Any]

    embedding: Optional[List[float]] = None

    usage_count: int = 0        
    success_count: int = 0     
    priority: float = 1.0       

    version_group_id: str = ""
    version: int = 1
    status: str = "active"    
    valid_from: str = ""
    valid_to: str = ""
    supersedes: str = ""       
    conflict_type: str = ""    
    last_hit_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def build_tree(if_conditions: List[str], then_steps: List[str]) -> Dict[str, Any]:
        return {
            "type": "sequence",
            "steps": [
                {"type": "action", "text": step}
                for step in then_steps
                if str(step).strip()
            ]
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ProcedureRule":
        if_conditions = data.get("if_conditions")
        if if_conditions is None:
            if_conditions = data.get("if_condition", [])
        if not isinstance(if_conditions, list):
            if_conditions = [str(if_conditions)]

        then_steps = data.get("then_steps", [])
        if not isinstance(then_steps, list):
            then_steps = [str(then_steps)]

        tree = data.get("tree")
        if not isinstance(tree, dict):
            tree = ProcedureRule.build_tree(if_conditions, then_steps)

        embedding = data.get("embedding")

        return ProcedureRule(
            id=data.get("id", "") or uuid.uuid4().hex,
            title=data.get("title", ""),
            if_conditions=if_conditions,
            then_steps=then_steps,
            tree=tree,
            tags=data.get("tags", []),
            source_text=data.get("source_text", ""),
            meta=data.get("meta", {}),
            embedding=embedding,
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            priority=data.get("priority", 1.0),
            version_group_id=data.get("version_group_id", ""),
            version=data.get("version", 1),
            status=data.get("status", "active"),
            valid_from=data.get("valid_from", ""),
            valid_to=data.get("valid_to", ""),
            supersedes=data.get("supersedes", ""),
            conflict_type=data.get("conflict_type", ""),
            last_hit_at=data.get("last_hit_at", ""),
        )



class EmbeddingHelper:
    _model = None
    _model_name = "distiluse-base-multilingual-cased-v2"

    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(cls._model_name, device='cpu')
        return cls._model

    @classmethod
    def encode(cls, text: str) -> List[float]:
        model = cls._get_model()
        vec = model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        return vec.tolist()


def _cosine_sim(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    n1 = 0.0
    n2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        n1 += a * a
        n2 += b * b
    if n1 == 0.0 or n2 == 0.0:
        return 0.0
    return dot / math.sqrt(n1 * n2)

class ProceduralMemoryStore:

    def __init__(
        self,
        storage_path: Optional[Path] = None,
        semantic_threshold: float = 0.85,
        namespace: str = "default"
    ):
        self.namespace = namespace

        if storage_path is None:
            base_dir = Path(__file__).parent.parent / "graph" / "procedural_memory" / namespace
            storage_path = base_dir / "procedural_rules.jsonl"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.semantic_threshold = semantic_threshold
        self.if_conflict_threshold = 0.8
        self.then_conflict_threshold = 0.45

        self._index: Dict[str, str] = {}     # key -> rule.id
        self._embeddings: Dict[str, List[float]] = {}

        self._loaded: bool = False

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _tokenize_text(text: str) -> set:
        text = (text or "").lower().strip()
        if not text:
            return set()
        for ch in [",", ".", "，", "。", ";", "；", ":", "：", "(", ")", "（", "）", "[", "]", "{", "}", "!", "?", "！", "？"]:
            text = text.replace(ch, " ")
        return {tok for tok in text.split() if tok}

    def _text_jaccard(self, t1: str, t2: str) -> float:
        s1 = self._tokenize_text(t1)
        s2 = self._tokenize_text(t2)
        if not s1 or not s2:
            return 0.0
        inter = len(s1.intersection(s2))
        union = len(s1.union(s2))
        return inter / union if union > 0 else 0.0

    def _rule_if_similarity(self, r1: ProcedureRule, r2: ProcedureRule) -> float:
        t1 = " ".join(r1.if_conditions or [])
        t2 = " ".join(r2.if_conditions or [])
        return self._text_jaccard(t1, t2)

    def _rule_then_similarity(self, r1: ProcedureRule, r2: ProcedureRule) -> float:
        t1 = " ".join(r1.then_steps or [])
        t2 = " ".join(r2.then_steps or [])
        return self._text_jaccard(t1, t2)

    def _detect_conflicting_rule(
        self,
        new_rule: ProcedureRule,
        rule_map: Dict[str, ProcedureRule],
    ) -> Optional[ProcedureRule]:
        best_rule = None
        best_if_sim = 0.0

        for old_rule in rule_map.values():
            if old_rule.status != "active":
                continue

            if_sim = self._rule_if_similarity(new_rule, old_rule)
            if if_sim < self.if_conflict_threshold:
                continue

            then_sim = self._rule_then_similarity(new_rule, old_rule)
            if then_sim <= self.then_conflict_threshold and if_sim > best_if_sim:
                best_if_sim = if_sim
                best_rule = old_rule

        return best_rule

    def _apply_conflict_versioning(
        self,
        new_rule: ProcedureRule,
        old_rule: ProcedureRule,
    ) -> None:
        now = self._now_iso()

        if not old_rule.version_group_id:
            old_rule.version_group_id = f"vg_{old_rule.id[:8]}"
        if not old_rule.valid_from:
            old_rule.valid_from = old_rule.last_hit_at or now

        old_rule.status = "deprecated"
        old_rule.valid_to = now

        if not new_rule.id:
            new_rule.id = uuid.uuid4().hex
        new_rule.version_group_id = old_rule.version_group_id
        new_rule.version = max(old_rule.version + 1, 1)
        new_rule.status = "active"
        new_rule.valid_from = now
        new_rule.valid_to = ""
        new_rule.supersedes = old_rule.id
        new_rule.conflict_type = "hard_contradiction"

    def _load_if_needed(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        if not self.storage_path.exists():
            return

        with self.storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    rule = ProcedureRule.from_dict(data)
                except Exception:
                    continue

                key = self._make_key(rule.if_conditions, rule.then_steps)
                self._index[key] = rule.id

                if isinstance(rule.embedding, list) and rule.embedding:
                    self._embeddings[rule.id] = rule.embedding


    @staticmethod
    def _make_key(if_conditions: List[str], then_steps: List[str]) -> str:
        norm_if = "|".join(s.strip().lower() for s in if_conditions if s.strip())
        norm_then = "|".join(s.strip().lower() for s in then_steps if s.strip())
        return norm_if + "||" + norm_then

    @staticmethod
    def _build_canonical_text(if_conditions: List[str], then_steps: List[str]) -> str:
        cond = "；".join(s.strip() for s in if_conditions if s.strip()) or "(Unconditional)"
        steps = "；".join(s.strip() for s in then_steps if s.strip()) or "(No subsequent steps)"
        return f"IF: {cond} THEN: {steps}"


    def _find_semantic_duplicate(
        self,
        emb: List[float],
        threshold: float = 0.85,
    ) -> Optional[str]:
        if not self._embeddings:
            return None

        best_id: Optional[str] = None
        best_sim: float = 0.0

        for rid, e in self._embeddings.items():
            sim = _cosine_sim(emb, e)
            if sim > best_sim:
                best_sim = sim
                best_id = rid

        if best_id is not None and best_sim >= threshold:
            return best_id

        return None

    def add_rules(self, rules: List[ProcedureRule]) -> List[str]:

        self._load_if_needed()
        rule_map = self._load_all_rules()

        result_ids: List[str] = []
        changed = False

        for rule in rules:
            key = self._make_key(rule.if_conditions, rule.then_steps)
            if key in self._index:
                existing_id = self._index[key]
                result_ids.append(existing_id)
                continue

            canonical_text = self._build_canonical_text(
                rule.if_conditions, rule.then_steps
            )
            emb = EmbeddingHelper.encode(canonical_text)

            dup_id = self._find_semantic_duplicate(emb)
            if dup_id is not None:
                self._index[key] = dup_id
                result_ids.append(dup_id)
                continue

            if not rule.id:
                rule.id = uuid.uuid4().hex

            conflicting_rule = self._detect_conflicting_rule(rule, rule_map)
            if conflicting_rule is not None:
                self._apply_conflict_versioning(rule, conflicting_rule)
                changed = True
            else:
                if not rule.version_group_id:
                    rule.version_group_id = f"vg_{rule.id[:8]}"
                rule.version = max(rule.version, 1)
                rule.status = "active"
                if not rule.valid_from:
                    rule.valid_from = self._now_iso()

            rule.embedding = emb
            self._index[key] = rule.id
            self._embeddings[rule.id] = emb
            rule_map[rule.id] = rule
            result_ids.append(rule.id)
            changed = True

        if changed:
            self._rewrite_all_rules(rule_map)
            if hasattr(self, '_rule_cache'):
                delattr(self, '_rule_cache')
            if hasattr(self, '_cache_mtime'):
                delattr(self, '_cache_mtime')

        return result_ids

    def _append_to_file(serules: List[ProcedureRule]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("a", encoding="utf-8") as f:
            for rule in rulf, les:
                f.write(json.dumps(rule.to_dict(), ensure_ascii=False) + "\n")

    def _load_all_rules(self) -> Dict[str, ProcedureRule]:
        if hasattr(self, '_rule_cache') and hasattr(self, '_cache_mtime'):
            if self.storage_path.exists():
                current_mtime = self.storage_path.stat().st_mtime
                if current_mtime == self._cache_mtime:
                    return self._rule_cache

        rule_map = {}
        if self.storage_path.exists():
            with self.storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    rule = ProcedureRule.from_dict(data)
                    rule_map[rule.id] = rule
        
        self._rule_cache = rule_map
        if self.storage_path.exists():
            self._cache_mtime = self.storage_path.stat().st_mtime
        
        return rule_map

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Tuple[ProcedureRule, float]]:
        self._load_if_needed()
        rule_map = self._load_all_rules()
        
        if not rule_map:
            return []
        
        query_emb = EmbeddingHelper.encode(query)

        raw_results: List[Tuple[ProcedureRule, float]] = []
        for rule in rule_map.values():
            if rule.embedding:
                sim = _cosine_sim(query_emb, rule.embedding)
                if sim >= threshold:
                    if rule.status == "active":
                        weighted = sim
                    elif rule.status == "deprecated":
                        weighted = sim * 0.6
                    elif rule.status == "archived":
                        weighted = sim * 0.3
                    else:
                        weighted = sim * 0.5
                    raw_results.append((rule, weighted))

        grouped: Dict[str, Tuple[ProcedureRule, float]] = {}
        for rule, score in raw_results:
            group_id = rule.version_group_id or rule.id
            if group_id not in grouped or score > grouped[group_id][1]:
                grouped[group_id] = (rule, score)

        results = list(grouped.values())

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_step(
        self,
        step_query: str,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[Tuple[ProcedureRule, str, int]]:

        self._load_if_needed()
        rule_map = self._load_all_rules()
        
        if not rule_map:
            return []
        
        query_emb = EmbeddingHelper.encode(step_query)
        results = []
        
        for rule in rule_map.values():
            for idx, step in enumerate(rule.then_steps):
                step_emb = EmbeddingHelper.encode(step)
                sim = _cosine_sim(query_emb, step_emb)
                if sim >= threshold:
                    results.append((rule, step, idx, sim))
        results.sort(key=lambda x: x[3], reverse=True)
        return [(r, s, i) for r, s, i, _ in results[:top_k]]

    def search_by_condition(
        self,
        condition: str,
        top_k: int = 10,
        threshold: float = 0.7
    ) -> List[ProcedureRule]:
        self._load_if_needed()
        rule_map = self._load_all_rules()
        
        if not rule_map:
            return []
        
        query_emb = EmbeddingHelper.encode(condition)
        results = []
        
        for rule in rule_map.values():
            best_sim = 0.0
            for cond in rule.if_conditions:
                cond_emb = EmbeddingHelper.encode(cond)
                sim = _cosine_sim(query_emb, cond_emb)
                best_sim = max(best_sim, sim)
            
            if best_sim >= threshold:
                results.append((rule, best_sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return [r for r, _ in results[:top_k]]

    def record_usage(self, rule_id: str, success: bool = True) -> None:

        rule_map = self._load_all_rules()
        rule = rule_map.get(rule_id)
        
        if not rule:
            return

        rule.usage_count += 1
        rule.last_hit_at = self._now_iso()
        if success:
            rule.success_count += 1

        self._rewrite_all_rules(rule_map)

        if hasattr(self, '_rule_cache'):
            delattr(self, '_rule_cache')

    def _rewrite_all_rules(self, rule_map: Dict[str, ProcedureRule]) -> None:

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.storage_path.open("w", encoding="utf-8") as f:
            for rule in rule_map.values():
                f.write(json.dumps(rule.to_dict(), ensure_ascii=False) + "\n")

    def get_top_rules(
        self,
        top_k: int = 10,
        sort_by: str = "usage"
    ) -> List[ProcedureRule]:

        self._load_if_needed()
        rule_map = self._load_all_rules()
        rules = list(rule_map.values())
        
        if sort_by == "usage":
            rules.sort(key=lambda r: r.usage_count, reverse=True)
        elif sort_by == "success_rate":
            rules.sort(
                key=lambda r: (
                    r.success_count / max(r.usage_count, 1),
                    r.usage_count
                ),
                reverse=True
            )
        elif sort_by == "priority":
            rules.sort(key=lambda r: r.priority, reverse=True)
        
        return rules[:top_k]

    def search(self, keyword: str, limit: int = 20) -> List[ProcedureRule]:
        self._load_if_needed()
        results: List[ProcedureRule] = []

        if not self.storage_path.exists():
            return results

        keyword = keyword.strip().lower()
        with self.storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                rule = ProcedureRule.from_dict(data)
                haystack = " ".join(
                    rule.if_conditions + rule.then_steps + [rule.title, rule.source_text]
                ).lower()
                if keyword in haystack:
                    results.append(rule)

        return results
