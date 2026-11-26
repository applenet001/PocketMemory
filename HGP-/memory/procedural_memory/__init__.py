from .procedural_memory import store_procedural_memory, extract_procedure_rules
from .store import (
    ProcedureRule,
    ProceduralMemoryStore,
    EmbeddingHelper,
)
_default_store = None


def _get_store() -> ProceduralMemoryStore:
    global _default_store
    if _default_store is None:
        _default_store = ProceduralMemoryStore()
    return _default_store


def query_procedural_memory(
    query: str,
    limit: int = 10,
    method: str = "vector",
    threshold: float = 0.5,
):
    store = _get_store()
    
    if method == "vector":
        results = store.vector_search(query, top_k=limit, threshold=threshold)
        return [
            {
                "rule": rule.to_dict(),
                "score": score,
            }
            for rule, score in results
        ]
    
    elif method == "keyword":
        rules = store.search(query, limit=limit)
        return [{"rule": r.to_dict()} for r in rules]
    
    elif method == "condition":
        rules = store.search_by_condition(query, top_k=limit, threshold=threshold)
        return [{"rule": r.to_dict()} for r in rules]
    
    elif method == "step":
        results = store.search_by_step(query, top_k=limit, threshold=threshold)
        return [
            {
                "rule": rule.to_dict(),
                "matched_step": step,
                "step_index": idx,
            }
            for rule, step, idx in results
        ]
    
    else:
        return query_procedural_memory(query, limit, method="vector", threshold=threshold)


def get_top_procedures(top_k: int = 10, sort_by: str = "usage"):
    store = _get_store()
    rules = store.get_top_rules(top_k=top_k, sort_by=sort_by)
    return [r.to_dict() for r in rules]


def record_procedure_usage(rule_id: str, success: bool = True):
    store = _get_store()
    store.record_usage(rule_id, success=success)


__all__ = [
    "store_procedural_memory",
    "extract_procedure_rules",
    "query_procedural_memory",
    "get_top_procedures",
    "record_procedure_usage",
    "ProcedureRule",
    "ProceduralMemoryStore",
    "EmbeddingHelper",
]