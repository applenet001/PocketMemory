
import argparse
import sys
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


load_dotenv(BASE_DIR / ".env")

# 1. memory classifier
from train.main import InferencePipeline

# 2. memory storage interfaces
from memory.semantic_memory import store_semantic_profile
from memory.episodic_memory import (
    store_episodic_memory,
    extract_to_semantic,
    extract_to_procedural,
    extract_state_from_episodes,
)
from memory.procedural_memory import store_procedural_memory
from memory.working_memory import WorkingMemory

# 3. retrieval interfaces
from memory.semantic_memory import query_semantic_profile, filter_by_vector_similarity
from memory.episodic_memory import query_episodic_memory
from memory.procedural_memory import query_procedural_memory

_classifier: Optional[InferencePipeline] = None
_working_memory: Optional[WorkingMemory] = None


def get_classifier() -> InferencePipeline:
    global _classifier
    if _classifier is None:
        _classifier = InferencePipeline()
    return _classifier


def get_working_memory(task_name: str = "main_task") -> WorkingMemory:
    global _working_memory
    if _working_memory is None:
        _working_memory = WorkingMemory(task_name=task_name)
    return _working_memory


def unified_search(
    query: str,
    scope: str = "all",
    limit: int = 5,
    user_id: str = "default",
    namespace: str = "default",
    verbose: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:

    results = {}

    if scope in ["all", "semantic"]:
        try:
            semantic_raw = query_semantic_profile(
                user_id=user_id,
                namespace=namespace,
                limit=max(limit * 5, 50),
            )
            semantic_results = filter_by_vector_similarity(
                triplets=semantic_raw,
                query=query,
                top_k=limit,
                threshold=0.15,
            ) if query and query.strip() else semantic_raw[:limit]
            results["semantic"] = semantic_results
        except Exception:
            results["semantic"] = []

    if scope in ["all", "episodic"]:
        try:
            episodic_results = query_episodic_memory(
                query=query,
                limit=limit,
                user_id=user_id,
                namespace=namespace,
            )
            results["episodic"] = episodic_results
        except Exception:
            results["episodic"] = []

    if scope in ["all", "procedural"]:
        try:
            procedural_results = query_procedural_memory(query, limit=limit)
            results["procedural"] = procedural_results
        except Exception:
            results["procedural"] = []
 
    if scope in ["all", "working"]:
        try:
            wm = get_working_memory()
            current_state = wm.get_current_state()
            results["working"] = current_state if current_state else {}
        except Exception:
            results["working"] = {}
    
    return results

def process_input(
    text: str,
    enable_semantic: bool = True,
    enable_episodic: bool = True,
    enable_procedural: bool = True,
    enable_working: bool = True,
    user_id: str = "default",
    namespace: str = "default",
) -> Dict[str, Any]:
    
    result = {
        "input": text,
        "timestamp": datetime.now().isoformat(),
        "stages": {},
    }

    classifier = get_classifier()
    classification = classifier.infer(text)
    
    result["stages"]["classification"] = classification
    
    labels = classification.get("labels", [])
    is_semantic = enable_semantic and "semantic" in labels
    is_episodic = enable_episodic and "episodic" in labels
    is_procedural = enable_procedural and "procedural" in labels
    is_working = enable_working and "working" in labels
    
    if is_semantic:
        try:
            semantic_result = store_semantic_profile(
                text=text,
                user_id=user_id,
                namespace=namespace,
                evidence_episode_ids=None,
                use_llm=True,
            )
            result["stages"]["semantic"] = semantic_result
        except Exception as e:
            result["stages"]["semantic"] = {"status": "error", "error": str(e)}

    if is_episodic:
        try:
            episodic_state_result = extract_state_from_episodes(
                text=text,
                event_type="user_message",
                lookback_episodes=5,
                auto_update_working_memory=enable_working,
                namespace=namespace,
            )
            
            result["stages"]["episodic"] = {
                "success": True,
                "episode_id": episodic_state_result.get("episode_id"),
                "extracted_state": episodic_state_result.get("extracted_state"),
                "working_memory_updated": episodic_state_result.get("working_memory_update", {}).get("success", False),
            }
        except Exception as e:
            result["stages"]["episodic"] = {"success": False, "error": str(e)}
    

    if is_procedural:
        try:
            procedural_result = store_procedural_memory(
                text=text,
                meta={"classification": classification}
            )
            result["stages"]["procedural"] = procedural_result
        except Exception as e:
            result["stages"]["procedural"] = {"error": str(e)}
    
    return result


def batch_process(texts: List[str], **kwargs) -> List[Dict[str, Any]]:
    results = []
    for text in texts:
        result = process_input(text, **kwargs)
        results.append(result)
    return results


def trigger_extraction(
    extract_semantic: bool = True,
    extract_procedural: bool = True,
    batch_size: int = 10,
) -> Dict[str, Any]:
    results = {}
    
    if extract_semantic:
        semantic_result = extract_to_semantic(
            batch_size=batch_size,
            min_episodes=5,
            enable_consolidation=True,
        )
        results["semantic_extraction"] = semantic_result
    
    if extract_procedural:
        procedural_result = extract_to_procedural(
            batch_size=batch_size,
            enable_deduplication=True,
        )
        results["procedural_extraction"] = procedural_result
    
    return results


def search_memory(
    query: str,
    limit: int = 5,
    user_id: str = "default",
    namespace: str = "default",
) -> Dict[str, Any]:
    results = {}
    
    try:
        semantic_raw = query_semantic_profile(
            user_id=user_id,
            namespace=namespace,
            limit=max(limit * 5, 50),
        )
        semantic_results = filter_by_vector_similarity(
            triplets=semantic_raw,
            query=query,
            top_k=limit,
            threshold=0.15,
        ) if query and query.strip() else semantic_raw[:limit]
        results["semantic"] = semantic_results
    except Exception:
        results["semantic"] = []
    
    try:
        episodic_results = query_episodic_memory(
            query=query,
            method="vector",
            limit=limit,
            user_id=user_id,
            namespace=namespace,
        )
        results["episodic"] = episodic_results.get("results", [])
    except Exception:
        results["episodic"] = []
    
    try:
        procedural_results = query_procedural_memory(query, limit=limit)
        results["procedural"] = procedural_results
    except Exception:
        results["procedural"] = []
    
    return results

def main():
    parser = argparse.ArgumentParser(description="HGP memory manager system")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["process", "search", "extract"],
        default="process",
    )
    args = parser.parse_args()
    
    if args.mode == "process":
        result = process_input(args.input)
    elif args.mode == "search":
        result = search_memory(args.input)
    elif args.mode == "extract":
        result = trigger_extraction()
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
