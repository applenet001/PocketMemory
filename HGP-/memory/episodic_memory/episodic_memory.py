import os
import sys
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))

env_path = BASE_DIR / ".env"
load_dotenv(env_path)

from .episode_store import (
    EpisodeStore,
    Episode,
    EventType,
    ExecutionStatus,
    EmbeddingHelper,
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_LLM_MODEL = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")

def _call_llm(prompt: str, system_prompt: str = "") -> str:
    try:
        from llm_api import call_deepseek
        return call_deepseek(prompt, system_prompt)
    except ImportError:
        from openai import OpenAI
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=DEEPSEEK_LLM_MODEL,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content


_STORES: Dict[Tuple[str, str], EpisodeStore] = {} 
_EPISODE_COUNTERS: Dict[Tuple[str, str], int] = {}  


def _get_store(user_id: str = "default", namespace: str = "default") -> EpisodeStore:
    key = (user_id, namespace)
    if key not in _STORES:
        _STORES[key] = EpisodeStore(namespace=namespace)
    if key not in _EPISODE_COUNTERS:
        _EPISODE_COUNTERS[key] = 0
    return _STORES[key]


def _increment_counter(user_id: str, namespace: str) -> int:
    key = (user_id, namespace)
    if key not in _EPISODE_COUNTERS:
        _EPISODE_COUNTERS[key] = 0
    _EPISODE_COUNTERS[key] += 1
    return _EPISODE_COUNTERS[key]


def _should_trigger_semantic_extraction(
    user_id: str, 
    namespace: str, 
    threshold: int = 5
) -> bool:

    key = (user_id, namespace)
    count = _EPISODE_COUNTERS.get(key, 0)
    
    if count >= threshold:

        _EPISODE_COUNTERS[key] = 0
        return True
    return False

def store_episodic_memory(
    text: str,
    event_type: str = "user_message",
    actors: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    use_llm_extract: bool = True,
    user_id: str = "default",
    namespace: str = "default",
    auto_extract_semantic: bool = True,  
    semantic_threshold: int = 5,  
) -> Dict[str, Any]:

    store = _get_store(user_id, namespace)

    if use_llm_extract:
        extracted = _extract_episode_structure(text)
    timestamp = None
    if metadata:
        if "session_date_time" in metadata and metadata["session_date_time"]:
            timestamp = metadata["session_date_time"]
        elif "timestamp" in metadata:
            timestamp = metadata["timestamp"]

    episode = Episode(
        event_type=event_type,
        content=text,
        actors=extracted.get("actors", actors or ["user"]),
        context=context or {},
        metadata=metadata or {"source": "classifier"},
        timestamp=timestamp, 
    )
    

    episode_id = store.add_episode(episode)
    count = _increment_counter(user_id, namespace)

    semantic_extracted = False
    if auto_extract_semantic and _should_trigger_semantic_extraction(user_id, namespace, semantic_threshold):

        from memory.semantic_memory import store_semantic_profile

        recent_episodes = store.query_recent_episodes(limit=semantic_threshold)

        combined_text = "\n\n".join([ep.get("content", "") for ep in recent_episodes])
        evidence_ids = [ep.get("id", "") for ep in recent_episodes]

        from memory.core.llm_extractor import SchemaBasedExtractor
        from memory.core.llm_config import LLMConfig
            
        api_key, base_url, model = LLMConfig.get_llm_config()
        extractor = SchemaBasedExtractor(api_key, base_url, model)
        extracted = extractor.extract_semantic_info(combined_text)

        features_to_store = []
            
        for feat in extracted.get("features", []):
                content = feat.get("content", feat.get("description", ""))
                if ":" in content:
                    predicate, obj = content.split(":", 1)
                    features_to_store.append({
                        "predicate": predicate.strip(),
                        "object": obj.strip(),
                        "confidence": 0.8,
                        "evidence_episode_ids": evidence_ids
                    })

        for like in extracted.get("likes", []):
                features_to_store.append({
                    "predicate": "likes",
                    "object": like.get("name", ""),
                    "confidence": 0.9,
                    "evidence_episode_ids": evidence_ids
                })
            
        for dislike in extracted.get("dislikes", []):
                features_to_store.append({
                    "predicate": "dislikes",
                    "object": dislike.get("name", ""),
                    "confidence": 0.9,
                    "evidence_episode_ids": evidence_ids
                })

        if features_to_store:
                result = store_hierarchical_features(
                    user_id=user_id,
                    features=features_to_store,
                    namespace=namespace
                )
                semantic_extracted = True
            
    
    return {
        "success": True,
        "episode_id": episode_id,
        "summary": episode.content[:100] if episode.content else "",  
        "episode_count": count,
        "semantic_extracted": semantic_extracted,
    }


def store_task_execution(
    task_description: str,
    action_sequence: List[str],
    success: bool,
    user_id: str = "default",
    namespace: str = "default",
    context: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:

    store = _get_store()

    if metadata is None:
        metadata = {}
    metadata["user_id"] = user_id
    metadata["namespace"] = namespace
    
    episode_id = store.add_task_execution(
        task_description=task_description,
        action_sequence=action_sequence,
        success=success,
        context=context,
        metadata=metadata,
    )
    
    return {
        "success": True,
        "episode_id": episode_id,
        "task": task_description,
        "steps": action_sequence,
        "execution_success": success,
        "user_id": user_id,
        "namespace": namespace,
    }


def _extract_episode_structure(text: str) -> Dict[str, Any]:
    
    prompt = f"""Please extract event information from the text below and return it in JSON format:

Text:
{text}

Please extract:
1. summary: Event summary (concise description, 20 characters or fewer)
2. actors: List of participants (names of people or entities)
3. event_time: Event time (if mentioned in the text)

Return only JSON; do not include any other content. Example format:
{{“summary”: “Dinner with friends”, “actors”: [“User”, ‘Xiaoming’], “event_time”: null}}
"""
    
    try:
        response = _call_llm(prompt)
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"summary": text[:50], "actors": ["user"]}
    except Exception:
        return {"summary": text[:50], "actors": ["user"]}

def query_episodic_memory(
    query: str,
    method: str = "vector",
    limit: int = 10,
    user_id: str = "default",
    namespace: str = "default",
    **kwargs
) -> Dict[str, Any]:
    store = _get_store(user_id=user_id, namespace=namespace)
    
    if method == "hybrid":
        threshold = kwargs.get("threshold", 0.5)
        use_graph_first = kwargs.get("use_graph_first", True)
        results = store.hybrid_search(query, top_k=limit, threshold=threshold, use_graph_first=use_graph_first)
        return {
            "method": "hybrid",
            "query": query,
            "results": [
                {
                    "episode_id": ep.id,
                    "summary": ep.content[:100] if ep.content else "",
                    "content": ep.content,
                    "timestamp": ep.timestamp,
                    "event_type": ep.event_type,
                    "score": score,
                }
                for ep, score in results
            ],
        }
    
    elif method == "vector":
        threshold = kwargs.get("threshold", 0.5)
        results = store.vector_search(query, top_k=limit, threshold=threshold)
        return {
            "method": "vector",
            "query": query,
            "results": [
                {
                    "episode_id": ep.id,
                    "summary": ep.content[:100] if ep.content else "",
                    "content": ep.content,
                    "timestamp": ep.timestamp,
                    "event_type": ep.event_type,
                    "score": score,
                }
                for ep, score in results
            ],
        }
    
    elif method == "timeline":
        start_time = kwargs.get("start_time")
        end_time = kwargs.get("end_time")
        event_type = kwargs.get("event_type")
        episodes = store.query_timeline(start_time, end_time, event_type, limit)
        return {
            "method": "timeline",
            "results": [
                {
                    "episode_id": ep.id,
                    "summary": ep.content[:100] if ep.content else "",
                    "timestamp": ep.timestamp,
                    "event_type": ep.event_type,
                }
                for ep in episodes
            ],
        }
    
    elif method == "actor":
        actor_name = kwargs.get("actor_name", query)
        episodes = store.query_by_actor(actor_name, limit)
        return {
            "method": "actor",
            "actor": actor_name,
            "results": [
                {
                    "episode_id": ep.id,
                    "summary": ep.content[:100] if ep.content else "",
                    "timestamp": ep.timestamp,
                }
                for ep in episodes
            ],
        }
    
    elif method == "causal":
        episode_id = kwargs.get("episode_id", query)
        direction = kwargs.get("direction", "forward")
        episodes = store.query_causal_chain(episode_id, direction, limit)
        return {
            "method": "causal",
            "start_episode": episode_id,
            "direction": direction,
            "chain": [
                {
                    "episode_id": ep.id,
                    "summary": ep.content[:100] if ep.content else "",
                    "timestamp": ep.timestamp,
                }
                for ep in episodes
            ],
        }
    
    else:
        return {"error": f"Unknown method: {method}"}


def extract_to_semantic(
    batch_size: int = 10,
    min_episodes: int = 5,
    enable_consolidation: bool = True,
    user_id: str = "default",
    namespace: str = "default",
) -> Dict[str, Any]:
    store = _get_store(user_id=user_id, namespace=namespace)

    episodes = store.get_unextracted_for_semantic(batch_size)
    
    if len(episodes) < min_episodes:
        return {
            "success": False,
            "message": f"事件数量不足（需要{min_episodes}条，当前{len(episodes)}条）",
        }

    episode_texts = []
    for ep in episodes:
        clean_content = ep.content
        if clean_content.startswith("【"):
            end_bracket = clean_content.find("】")
            if end_bracket > 0:
                clean_content = clean_content[end_bracket + 1:]

        episode_texts.append(f"- {clean_content[:200]}")
    
    combined_text = "\n".join(episode_texts)
    
    extraction_prompt = f"""Analyze the historical events of the following user to extract **timeless** user profiles and knowledge:

Event list:
{combined_text}

**Extraction Rules (Important):**
1. Extract only stable characteristics, preferences, values, and knowledge
2. Do not include specific times, dates, or frequencies (e.g., “August,” “last week,” “3 times”)
3. Do not include full dialogue content or timestamps
4. Use third-person statements; be concise and clear

**Examples:**
✓ “The user supports LGBTQ rights”
✓ “The user likes to express themselves through art”
✓ “The user plans to adopt a child”
✓ “The user values the support of family and friends”
✗ “The user participated in a march in August” (includes a specific time → this is episodic memory)
✗ “[2:24 pm] User: ...” (includes the original dialogue → not needed)
✗ “The user painted a picture last week” (includes a time → this is episodic memory)

Please extract:
1. traits: Identity traits, personality traits (e.g., “is a transgender woman,” “outgoing and cheerful”)
2. preferences: Preferences and interests (e.g., “enjoys pottery,” “enjoys outdoor activities”)
3. values: Values and beliefs (e.g., “believes art promotes freedom,” “values equality”)
4. goals: Goals and plans (e.g., “wants to become a counselor,” “plans to adopt”)
5. knowledge: General knowledge (objective knowledge unrelated to the user)

Return JSON format:
{{“traits”: [“Trait 1”], “preferences”: [“Preference 1”], “values”: [“Value 1”], “goals”: [“Goal 1”], ‘knowledge’: [“Knowledge 1”]}}
"""
    
    try:
        response = _call_llm(extraction_prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = {"traits": [], "preferences": [], "values": [], "goals": [], "knowledge": []}
        new_features = []
        
        for trait in extracted.get("traits", []):
            new_features.append({
                "type": "trait",
                "content": trait,
                "source": "episodic_extract",
            })
        
        for pref in extracted.get("preferences", []):
            new_features.append({
                "type": "preference",
                "content": pref,
                "source": "episodic_extract",
            })
        
        for value in extracted.get("values", []):
            new_features.append({
                "type": "value",
                "content": value,
                "source": "episodic_extract",
            })
        
        for goal in extracted.get("goals", []):
            new_features.append({
                "type": "goal",
                "content": goal,
                "source": "episodic_extract",
            })
        
        for knowledge in extracted.get("knowledge", []):
            new_features.append({
                "type": "knowledge",
                "content": knowledge,
                "source": "episodic_extract",
            })
        if not new_features:
            return {
                "success": True,
                "episodes_processed": len(episodes),
                "message": "No new semantic features were extracted.",
                "new_features": 0,
            }

        if enable_consolidation:
            consolidated_result = _consolidate_semantic_features(new_features)
            commands = consolidated_result.get("commands", {})
        else:
            commands = {
                "add": new_features,
                "update": [],
                "delete": [],
                "skip": [],
            }

        execution_result = _execute_semantic_commands(commands)

        for ep in episodes:
            store.mark_extracted(ep.id, "semantic")
        
        return {
            "success": True,
            "episodes_processed": len(episodes),
            "new_features_extracted": len(new_features),
            "commands": {
                "added": len(commands.get("add", [])),
                "updated": len(commands.get("update", [])),
                "deleted": len(commands.get("delete", [])),
                "skipped": len(commands.get("skip", [])),
            },
            "execution": execution_result,
            "consolidation_enabled": enable_consolidation,
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _consolidate_semantic_features(new_features: List[Dict]) -> Dict[str, Any]:

    try:
        from memory.semantic_memory import query_semantic_profile
        existing_features = query_semantic_profile(
            user_id="default",
            namespace="default",
            limit=500,
        )
        
        if not existing_features:
            return {
                "commands": {
                    "add": new_features,
                    "update": [],
                    "delete": [],
                    "skip": [],
                }
            }

        consolidation_prompt = f"""Your task is to perform semantic memory consolidation.

Consolidation Principles (based on cognitive science):
1. **Minimize Interference**: Remove redundant information to prevent interference between memories
2. **Synchronize Similar Memories**: Apply a unified format and structure to similar features
3. **Merge duplicate information**: Merge multiple memories expressing the same meaning into a single one
4. **Retain key differences**: Keep memories with differing details separate

Existing semantic memories:
{json.dumps(existing_features[:50], ensure_ascii=False, indent=2)}
(Total of {len(existing_features)} entries)

Newly extracted features:
{json.dumps(new_features, ensure_ascii=False, indent=2)}

Please determine how each new feature should be handled:

1. **add**: New information; should be added
   - Example: This preference/habit is not mentioned in existing memories
   
2. **update**: Update an existing memory
   - Example: “Likes coffee” updated to “Likes black coffee, no sugar”
   
3. **skip**: Completely redundant; skip
   - Example: The new memory “Likes coffee” already exists and contains no new information
   
4. **merge**: Merge multiple similar memories
   - Example: Merge “Likes coffee,” “Likes lattes,” and “Likes Americanos” into “Likes various types of coffee (lattes, Americanos)”

Return JSON format:
{{
    “add”: [{{type, content, source}}],
    “update”: [{{old_content, new_content, reason}}],
    “skip”: [{{content, reason}}],
    “merge”: [{{contents, merged_content}}]
}}

Notes:
- Err on the side of caution (skip) rather than merging excessively and causing information loss
- Only perform an update when there is sufficient evidence
- A merge requires at least 3 similar items
"""
        
        response = _call_llm(consolidation_prompt)
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            commands = json.loads(json_match.group())
        else:
            commands = {
                "add": new_features,
                "update": [],
                "skip": [],
                "merge": [],
            }
        if commands.get("merge"):
            for merge_item in commands["merge"]:
                commands["add"].append({
                    "type": "merged",
                    "content": merge_item.get("merged_content", ""),
                    "source": "consolidation_merge",
                })
        
        return {"commands": commands}
    
    except ImportError:
        return {
            "commands": {
                "add": new_features,
                "update": [],
                "skip": [],
            }
        }
    except Exception:
        return {
            "commands": {
                "add": new_features,
                "update": [],
                "skip": [],
            }
        }


def _execute_semantic_commands(commands: Dict[str, List]) -> Dict[str, Any]:
    try:
        from memory.semantic_memory import store_semantic_profile
        
        added_ids = []
        updated_ids = []

        for item in commands.get("add", []):
            content_text = item.get("content", "")
            result = store_semantic_profile(
                text=content_text,
                user_id="default",  
                namespace="default",  
                evidence_episode_ids=None,
                use_llm=True
            )
            if result.get("features_added", 0) > 0:
                added_ids.append(f"feature_{len(added_ids)+1}")

        
        return {
            "success": True,
            "added_count": len(added_ids),
            "updated_count": len(updated_ids),
            "added_ids": added_ids,
            "updated_ids": updated_ids,
        }
    
    except ImportError:
        return {
            "success": False,
            "error": "The semantic memory module is unavailable",
            "commands_queued": commands,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def extract_to_procedural(
    batch_size: int = 10,
    min_success_rate: float = 0.8,
    enable_deduplication: bool = True,
) -> Dict[str, Any]:
    store = _get_store()
    episodes = store.get_successful_task_sequences(batch_size)
    
    if not episodes:
        return {
            "success": False,
            "message": "No records of successfully completed tasks were found.",
        }

    candidate_rules = []
    
    for ep in episodes:
        if not ep.action_sequence:
            continue
        
        rule = {
            "title": ep.content[:100] if ep.content else "",
            "if_conditions": [f"需要{ep.content[:100] if ep.content else ''}"],
            "then_steps": ep.action_sequence,
            "source_text": ep.content,
            "meta": {
                "source": "episodic_extract",
                "original_episode_id": ep.id,
                "timestamp": ep.timestamp,
                "success": True,
            },
        }
        
        candidate_rules.append(rule)
    
    if not candidate_rules:
        return {
            "success": False,
            "message": "No records of successfully completed tasks were found.",
        }

    if enable_deduplication:
        dedup_result = _deduplicate_procedural_rules(candidate_rules)
        final_rules = dedup_result.get("unique_rules", [])
        duplicates_found = dedup_result.get("duplicates", [])
    else:
        final_rules = candidate_rules
        duplicates_found = []

    stored_ids = []
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "procedural_memory"))
        from procedural_memory import store_procedural_memory
        
        for rule in final_rules:
            result = store_procedural_memory(
                rule["source_text"],
                meta=rule["meta"],
            )
            if result.get("rules"):
                stored_ids.extend([r["id"] for r in result["rules"]])
    except ImportError:
        pass

    for ep in episodes:
        store.mark_extracted(ep.id, "procedural")
    
    return {
        "success": True,
        "episodes_processed": len(episodes),
        "candidate_rules": len(candidate_rules),
        "unique_rules": len(final_rules),
        "duplicates_skipped": len(duplicates_found),
        "stored_rule_ids": stored_ids,
        "rules": final_rules,
        "deduplication_enabled": enable_deduplication,
    }


def _deduplicate_procedural_rules(candidate_rules: List[Dict]) -> Dict[str, Any]:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "procedural_memory"))
        from procedural_memory import query_procedural_memory
        
        existing_rules = query_procedural_memory(
            query="",
            limit=200,
        )
        
        if not existing_rules:
            return {
                "unique_rules": candidate_rules,
                "duplicates": [],
            }

        dedup_prompt = f"""Your task is to determine whether the new procedural memory rule duplicates an existing rule.

Procedural memory rule format:
- title: Task title
- then_steps: List of steps to execute

Existing rules ({len(existing_rules)} rules):
{json.dumps([{{"title": r.get("title", ""), "steps": r.get("then_steps", [])}} for r in existing_rules[:20]], ensure_ascii=False, indent=2)}

New candidate rules:
{json.dumps([{{"title": r.get("title", ""), "steps": r.get("then_steps", [])}} for r in candidate_rules], ensure_ascii=False, indent=2)}

Evaluate each new rule:
1. **unique**: A completely new process; should be added
   - Example: No similar task steps in existing rules
   
2. **duplicate**: Highly duplicated with existing rules (>80% similarity); skip
   - Example: Steps are exactly the same or differ only in wording
   
3. **similar**: Partially duplicated but with differences; retain
   - Example: Core steps are the same but details differ

Return JSON:
{{
    “unique”: [{{index: 0, title: “...”}}],
    “duplicate”: [{{index: 1, title: “...”, similar_to: “Existing rule title”}}],
    “similar”: [{{index: 2, title: “...”, note: “Description of differences”}}]
}}
"""
        
        response = _call_llm(dedup_prompt)

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            dedup_result = json.loads(json_match.group())
        else:
            return {
                "unique_rules": candidate_rules,
                "duplicates": [],
            }
        
        unique_indices = set()
        
        for item in dedup_result.get("unique", []):
            unique_indices.add(item.get("index"))
        
        for item in dedup_result.get("similar", []):
            unique_indices.add(item.get("index"))
        
        unique_rules = [candidate_rules[i] for i in sorted(unique_indices) if i < len(candidate_rules)]
        
        duplicates = dedup_result.get("duplicate", [])
        
        return {
            "unique_rules": unique_rules if unique_rules else candidate_rules,
            "duplicates": duplicates,
        }
    
    except ImportError:
        return {
            "unique_rules": candidate_rules,
            "duplicates": [],
        }
    except Exception:
        return {
            "unique_rules": candidate_rules,
            "duplicates": [],
        }


def analyze_episode_patterns(
    time_range_days: int = 30,
    min_frequency: int = 3,
) -> Dict[str, Any]:
    store = _get_store()
    from datetime import datetime, timedelta
    end_time = datetime.now().isoformat()
    start_time = (datetime.now() - timedelta(days=time_range_days)).isoformat()
    
    episodes = store.query_timeline(start_time, end_time, limit=500)
    
    if not episodes:
        return {"patterns": [], "message": "There are not enough events to analyze"}

    event_type_counts = {}
    actor_counts = {}
    
    for ep in episodes:
        event_type_counts[ep.event_type] = event_type_counts.get(ep.event_type, 0) + 1
        
        for actor in ep.actors:
            actor_counts[actor] = actor_counts.get(actor, 0) + 1
    
    patterns = {
        "frequent_event_types": {
            k: v for k, v in event_type_counts.items() if v >= min_frequency
        },
        "frequent_actors": {
            k: v for k, v in actor_counts.items() if v >= min_frequency
        },
    }
    
    return {
        "time_range_days": time_range_days,
        "total_episodes": len(episodes),
        "patterns": patterns,
    }


def get_episodic_stats() -> Dict[str, Any]:
    store = _get_store()
    return store.get_stats()

def search_episodes(query: str, limit: int = 10) -> List[Dict]:
    result = query_episodic_memory(query, method="vector", limit=limit)
    return result.get("results", [])


def get_recent_episodes(limit: int = 20) -> List[Dict]:
    result = query_episodic_memory("", method="timeline", limit=limit)
    return result.get("results", [])


def extract_state_from_episodes(
    text: str,
    event_type: str = "user_message",
    lookback_episodes: int = 5,
    session_id: Optional[str] = None,
    auto_update_working_memory: bool = True,
    metadata: Optional[Dict[str, Any]] = None, 
    namespace: Optional[str] = None, 
) -> Dict[str, Any]:
    from .state_extractor import StateExtractor
    
    result = {
        "original_text": text,
        "timestamp": datetime.now().isoformat(),
        "episode_id": None,
        "extracted_state": None,
        "working_memory_update": None,
    }

    full_metadata = {"session_id": session_id or "default"}
    if metadata:
        full_metadata.update(metadata)

    if namespace:

        from .episode_store import EpisodeStore
        store = EpisodeStore(namespace=namespace)

        dataset_timestamp = full_metadata.get("session_date_time")
        if dataset_timestamp:
            episode_timestamp = dataset_timestamp
        else:
            episode_timestamp = datetime.now().isoformat()
        
        episode = Episode(
            event_type=event_type,
            content=text,
            actors=["user"],
            context={},
            metadata=full_metadata,
            timestamp=episode_timestamp, 
        )
        episode_id = store.add_episode(episode)
        result["episode_id"] = episode_id
    else:
        store_result = store_episodic_memory(
            text=text,
            event_type=event_type,
            metadata=full_metadata,
            use_llm_extract=False,
        )
        result["episode_id"] = store_result.get("episode_id")
        store = _get_store()
    recent_episodes_data = store.query_timeline(limit=lookback_episodes)

    recent_episodes = []
    for ep in recent_episodes_data:
        if isinstance(ep, Episode):
            recent_episodes.append(ep)
        elif isinstance(ep, dict):
            recent_episodes.append(Episode(
                id=ep.get("id", ""),
                timestamp=ep.get("timestamp", ""),
                event_type=ep.get("event_type", "user_message"),
                content=ep.get("content", ""),
                actors=ep.get("actors", []),
                execution_status=ep.get("execution_status", "unknown"),
            ))
    
    if recent_episodes:
        extracted_state = StateExtractor.extract_state(
            recent_episodes=recent_episodes,
            context={"session_id": session_id or "default", "event_type": event_type}
        )
        result["extracted_state"] = extracted_state.to_dict()

        if auto_update_working_memory:
            wm_update = _update_working_memory_with_state(
                extracted_state=extracted_state,
                session_id=session_id
            )
            result["working_memory_update"] = wm_update
    
    return result


def _update_working_memory_with_state(
    extracted_state,
    session_id: Optional[str] = None
) -> Dict[str, Any]:

    try:
        from memory.working_memory import WorkingMemory
        
        wm = WorkingMemory(task_name=f"session_{session_id or 'default'}")

        content = {
            "text": extracted_state.recent_episodes[-1].content if extracted_state.recent_episodes else "",
            "progress": f"{extracted_state.progress_step}/{extracted_state.total_steps}",
        }
        
        current = wm.get_current_state()
        if current:
            node_id = wm.transition_to(
                next_state_name=extracted_state.state_name,
                content=content,
                reason=f"Inferring from the sequence of events: {extracted_state.episode_chain_summary}",
                priority=extracted_state.priority,
                episode_id="episodic",
            )
        else:
            node_id = wm.start_state(
                state_name=extracted_state.state_name,
                content=content,
                episode_id="episodic",
                priority=extracted_state.priority,
            )
        
        return {
            "success": True,
            "node_id": node_id,
            "state_name": extracted_state.state_name,
            "priority": extracted_state.priority,
            "progress": f"{extracted_state.progress_step}/{extracted_state.total_steps}",
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
