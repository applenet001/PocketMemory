from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

_classifier = None
_working_memory_instances = {}


@dataclass
class MemoryConfig:

    storage_dir: Optional[Path] = None
    classifier_threshold: float = 0.9
    neo4j_enabled: bool = True
    enable_semantic: bool = True
    enable_episodic: bool = True
    enable_procedural: bool = True
    enable_working: bool = True
    default_top_k: int = 5
    evolution_enabled: bool = False
    evolution_threshold: int = 3  

    auto_extract_enabled: bool = True
    auto_extract_threshold: int = 20 
    auto_extract_async: bool = True  


@dataclass
class MemoryResult:
    success: bool
    memory_id: Optional[str] = None
    memory_types: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SearchResult:

    semantic: List[Dict[str, Any]] = field(default_factory=list)
    episodic: List[Dict[str, Any]] = field(default_factory=list)
    procedural: List[Dict[str, Any]] = field(default_factory=list)
    working: Optional[Dict[str, Any]] = None
    
    def to_context(
        self, query: str = "", max_items: int = 3, *,
        working_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:

        context_parts = []    

        if query:
            context_parts.append("[Current Task / Issue]")
            context_parts.append(query)
            context_parts.append("")
        if working_history is not None:
            context_parts.append("[Relevant Working States]")
            for i, item in enumerate(working_history[:max_items], 1):
                context_parts.append(f"State {i}（{item['timestamp']}）：{item['text'][:200]}")
            if not working_history:
                context_parts.append("No relevant working states")
        elif self.working:
            context_parts.append("[Current Working State]")
            state_name = self.working.get("state_name", "None")
            focus = self.working.get("content", {}).get("focus", "None")
            progress = self.working.get("content", {}).get("progress", "None")
            context_parts.append(f"State Name：{state_name}")
            context_parts.append(f"Current Focus：{focus}")
            context_parts.append(f"Progress：{progress}")
        else:
            context_parts.append("[Current Working State]")
            context_parts.append("State Name：None")
            context_parts.append("Current Focus：None")
            context_parts.append("Progress：None")
        context_parts.append("")

        context_parts.append("[Episodic Memory]")
        if self.episodic and len(self.episodic) > 0:
            for i, item in enumerate(self.episodic[:max_items], 1):
                summary = item.get("summary", "")
                content = item.get("content", "")
                timestamp = item.get("timestamp", item.get("when", ""))
                score = item.get("score")
                text = content if content else summary
                if not text:
                    continue
                
                if timestamp and score is not None:
                    context_parts.append(f"Episode {i}（{timestamp}, Similarity: {score:.2f}）：{text[:200]}")
                elif timestamp:
                    context_parts.append(f"Episode {i}（{timestamp}）：{text[:200]}")
                else:
                    context_parts.append(f"Episode {i}：{text[:200]}")
        else:
            context_parts.append("No relevant experience")
        context_parts.append("")

        context_parts.append("[Semantic Memory]")
        if self.semantic and len(self.semantic) > 0:
            for i, item in enumerate(self.semantic[:max_items], 1):
                if "relation" in item and "entity" in item:
                    relation = item.get("relation", "")
                    entity = item.get("entity", "")
                    display_text = f"{relation.replace('_', ' ')}: {entity}"
                else:
                    obj = item.get("object", "")
                    content = item.get("content", item.get("text", ""))
                    display_text = obj if obj else content
                
                if not display_text:
                    continue
                similarity = item.get("similarity")

                if similarity is not None:
                    context_parts.append(f"- {display_text[:100]} (相似度: {similarity:.2f})")
                else:
                    context_parts.append(f"- {display_text[:100]}")
        else:
            context_parts.append("")
        context_parts.append("")
        context_parts.append("[Procedural Memory]")
        if self.procedural and len(self.procedural) > 0:
            for i, item in enumerate(self.procedural[:max_items], 1):
                context_parts.append(format_procedural_rule(item.get("rule", item), i))
        else:
            context_parts.append("No rules available")
        context_parts.append("")        
        return "\n".join(context_parts)
    
    def is_empty(self) -> bool:
        return not self.semantic and not self.episodic and not self.procedural and not self.working


def format_procedural_rule(rule, number: int) -> str:
    if not isinstance(rule, dict):
        return f"Rule {number}：{str(rule)[:150]}"
    title = rule.get("title", f"Rule {number}")
    conditions = rule.get("if_conditions", [])
    steps = rule.get("then_steps", [])
    cond_str = ", ".join(conditions[:2]) if conditions else "Trigger conditions"
    step_str = " → ".join(steps[:2]) if steps else "Steps to Follow"
    return f"Rule {number}（{title}）：IF {cond_str} THEN {step_str}"


class Memory:   
    def __init__(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        session_id: Optional[str] = None,
        config: Optional[MemoryConfig] = None,
    ):

        self.user_id = user_id or "default"
        self.agent_id = agent_id
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.config = config or MemoryConfig()

        self.namespace = f"user_{self.user_id}"
        
        self._classifier = None
        self._working_memory = None
        self._episodic_store = None
        self._procedural_store = None

        self._metadata = {
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }
        
        self._episode_count_since_last_extract = 0
    
    def _get_classifier(self):
        if self._classifier is None:
            from train.main import InferencePipeline
            from train.adaptive_threshold import get_current_classifier_threshold
            current_threshold = get_current_classifier_threshold()
            self._classifier = InferencePipeline(threshold=current_threshold)
        return self._classifier
    
    def _get_working_memory(self):
        if self._working_memory is None:
            from memory.working_memory import WorkingMemory
            task_name = f"{self.user_id}_{self.session_id}"
            self._working_memory = WorkingMemory(task_name=task_name)
        return self._working_memory
    
    def _get_episodic_store(self):
        if self._episodic_store is None:
            from memory.episodic_memory.episode_store import EpisodeStore
            self._episodic_store = EpisodeStore(
                namespace=self.namespace,
                use_neo4j=self.config.neo4j_enabled
            )
        return self._episodic_store
    
    def _get_procedural_store(self):
        if self._procedural_store is None:
            from memory.procedural_memory.store import ProceduralMemoryStore
            self._procedural_store = ProceduralMemoryStore(
                namespace=self.namespace
            )
        return self._procedural_store
    
    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        metadata: Optional[Dict[str, Any]] = None,
        infer: bool = True,
        fast_mode: bool = False,
    ) -> MemoryResult:
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            text = "\n".join([
                f"{m.get('role', 'user')}: {m.get('content', '')}" 
                for m in messages if isinstance(m, dict)
            ])
        else:
            return MemoryResult(
                success=False,
                details={"error": f"Unsupported input types: {type(messages).__name__}"}
            )
        
        if not text.strip():
            return MemoryResult(
                success=False,
                details={"error": "The input is empty"}
            )
        
        full_metadata = {**self._metadata, **(metadata or {})}
        
        result_details = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "timestamp": datetime.now().isoformat(),
        }
        stored_types = []
        
        if infer:
            classifier = self._get_classifier()
            classification = classifier.infer(text)
            labels = classification.get("labels", [])
            result_details["classification"] = {
                "labels": labels,
                "source": classification.get("source"),
                "confidence": classification.get("max_confidence", 0),
            }
        else:
            labels = ["episodic"]
            result_details["classification"] = {"labels": labels, "source": "default"}

        if self.config.enable_semantic and "semantic" in labels:
            try:
                from memory.semantic_memory import store_semantic_profile

                semantic_result = store_semantic_profile(
                    text=text,
                    user_id=self.user_id,
                    namespace=self.namespace,
                    evidence_episode_ids=None,
                    use_llm=True
                )
                
                result_details["semantic"] = {
                    "success": True,
                    "triplets_added": semantic_result.get("triplets_added", 0),
                    "triplets_updated": semantic_result.get("triplets_updated", 0),
                    "triplets_skipped": semantic_result.get("triplets_skipped", 0),
                }
                stored_types.append("semantic")
            except Exception as e:
                result_details["semantic"] = {"success": False, "error": str(e)}
        
        if self.config.enable_episodic and "episodic" in labels:
            try:
                if fast_mode:
                    from memory.episodic_memory.episode_store import Episode
                    

                    store = self._get_episodic_store()

                    episode = Episode(
                        event_type="user_message",
                        content=text,
                        actors=["user"],
                        context={},
                        metadata=full_metadata,
                        timestamp=full_metadata.get("session_date_time"),  
                    )
                    episode_id = store.add_episode(episode)
                    
                    result_details["episodic"] = {
                        "success": True,
                        "episode_id": episode_id,
                        "mode": "fast",
                    }
                    stored_types.append("episodic")
                else:
                    from memory.episodic_memory import extract_state_from_episodes

                    episodic_result = extract_state_from_episodes(
                        text=text,
                        event_type="user_message",
                        lookback_episodes=5,
                        session_id=self.session_id,
                        auto_update_working_memory=self.config.enable_working,
                        metadata=full_metadata,  
                        namespace=self.namespace,  
                    )
                    
                    result_details["episodic"] = {
                        "success": True,
                        "episode_id": episodic_result.get("episode_id"),
                        "extracted_state": episodic_result.get("extracted_state"),
                    }
                    stored_types.append("episodic")
                    wm_update = episodic_result.get("working_memory_update")
                    if wm_update and isinstance(wm_update, dict) and wm_update.get("success"):
                        result_details["working"] = {
                            "success": True,
                            "node_id": wm_update.get("node_id"),
                            "source": "episodic_state_extraction",
                        }
                        if "working" not in stored_types:
                            stored_types.append("working")
                        
            except Exception as e:
                try:
                    from memory.episodic_memory import store_episodic_memory
                    episodic_result = store_episodic_memory(
                        text=text, 
                        metadata=full_metadata,
                        use_llm_extract=False if fast_mode else True
                    )
                    result_details["episodic"] = {
                        "success": True,
                        "episode_id": episodic_result.get("episode_id"),
                    }
                    stored_types.append("episodic")
                except Exception as e2:
                    result_details["episodic"] = {"success": False, "error": str(e2), "fallback_error": str(e)}
        
        if self.config.enable_procedural and "procedural" in labels:
            try:
                from memory.procedural_memory import store_procedural_memory
                procedural_result = store_procedural_memory(text=text, meta=full_metadata)
                result_details["procedural"] = {
                    "success": True,
                    "rule_count": len(procedural_result.get("rules", [])),
                    "stored_rule_ids": procedural_result.get("stored_rule_ids", []),
                }
                stored_types.append("procedural")
            except Exception as e:
                result_details["procedural"] = {"success": False, "error": str(e)}

        if self.config.enable_working and "working" in labels and "working" not in stored_types:
            try:
                wm = self._get_working_memory()
                node_id = wm.start_state(
                    state_name=text[:50],
                    content={"text": text, **full_metadata},
                )
                result_details["working"] = {"success": True, "node_id": node_id}
                stored_types.append("working")
            except Exception as e:
                result_details["working"] = {"success": False, "error": str(e)}
        
        if self.config.auto_extract_enabled and "episodic" in stored_types:
            self._episode_count_since_last_extract += 1
            
            if self._episode_count_since_last_extract >= self.config.auto_extract_threshold:
                if self.config.auto_extract_async:
                    import threading
                    threading.Thread(
                        target=self._auto_extract_semantic,
                        daemon=True
                    ).start()
                else:
                    self._auto_extract_semantic()
                
                self._episode_count_since_last_extract = 0
        
        return MemoryResult(
            success=len(stored_types) > 0 or len(labels) == 0,
            memory_types=stored_types,
            details=result_details,
        )
    
    def _auto_extract_semantic(self):
        try:
            from memory.episodic_memory import extract_to_semantic
            
            extract_to_semantic(
                batch_size=self.config.auto_extract_threshold,
                min_episodes=5,
                enable_consolidation=True,
                user_id=self.user_id,
                namespace=self.namespace,
            )
        except Exception:
            pass
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_types: Optional[List[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        semantic_mode: str = "smart",  # "smart" | "profile" | "vector"
    ) -> SearchResult:
        if memory_types is None:
            memory_types = ["semantic", "episodic", "procedural", "working"]
        
        result = SearchResult()

        if "semantic" in memory_types and self.config.enable_semantic:
            try:
                if semantic_mode == "profile":
                    from memory.semantic_memory import get_full_profile
                    profile_dict = get_full_profile(
                        user_id=self.user_id,
                        namespace=self.namespace
                    )
                    semantic_raw = []
                    for category, features in profile_dict.items():
                        for feat in features[:top_k]:
                            feat['category'] = category
                            semantic_raw.append(feat)
                    result.semantic = semantic_raw
                elif semantic_mode == "vector":
                    from memory.semantic_memory import query_semantic_profile, filter_by_vector_similarity
                    all_triplets = query_semantic_profile(
                        user_id=self.user_id,
                        namespace=self.namespace,
                        limit=100
                    )
                    if query and query.strip():
                        semantic_raw = filter_by_vector_similarity(
                            triplets=all_triplets,
                            query=query,
                            top_k=top_k,
                            threshold=0.15
                        )
                    else:
                        semantic_raw = all_triplets[:top_k]
                    result.semantic = semantic_raw if isinstance(semantic_raw, list) else []
                else:
                    from memory.semantic_memory import query_semantic_profile, filter_by_vector_similarity
                    all_triplets = query_semantic_profile(
                        user_id=self.user_id,
                        namespace=self.namespace,
                        limit=100
                    )
                    if query and query.strip():
                        semantic_raw = filter_by_vector_similarity(
                            triplets=all_triplets,
                            query=query,
                            top_k=top_k,
                            threshold=0.15
                        )
                    else:
                        semantic_raw = all_triplets[:top_k]
                    result.semantic = semantic_raw if isinstance(semantic_raw, list) else []
            except Exception:
                try:
                    from memory.semantic_memory import query_semantic_profile
                    semantic_raw = query_semantic_profile(
                        user_id=self.user_id,
                        namespace=self.namespace,
                        limit=top_k
                    )
                    result.semantic = semantic_raw if isinstance(semantic_raw, list) else []
                except Exception:
                    pass
        
        if "episodic" in memory_types and self.config.enable_episodic:
            try:
                store = self._get_episodic_store()
                episodes = store.hybrid_search(
                    query, 
                    top_k=top_k, 
                    threshold=0.02,
                    use_graph_first=True,
                    graph_weight=0.4,
                    vector_weight=0.6,
                    metadata_filter=metadata_filter
                )
                result.episodic = [
                    {
                        "episode_id": ep.id,
                        "summary": ep.content[:100] if ep.content else "",
                        "content": ep.content,
                        "timestamp": ep.timestamp,
                        "event_type": ep.event_type,
                        "score": score,
                    }
                    for ep, score in episodes
                ]
            except Exception:
                pass
        
        if "procedural" in memory_types and self.config.enable_procedural:
            try:
                store = self._get_procedural_store()
                rules = store.vector_search(query, top_k=top_k, threshold=0.3)
                result.procedural = [
                    {
                        "rule": rule.to_dict(),
                        "score": score,
                    }
                    for rule, score in rules
                ]
            except Exception:
                pass
        
        if "working" in memory_types and self.config.enable_working:
            try:
                wm = self._get_working_memory()
                summary = wm.get_summary()
                active_items = []
                
                active_states = summary.get("active_states", {})
                for node_id, node_data in active_states.items():
                    content_data = node_data.get("content", {})
                    active_items.append({
                        "content": content_data.get("text", str(content_data)),
                        "status": content_data.get("status", "未完成"),
                        "priority": node_data.get("priority", 5),
                        "timestamp": node_data.get("timestamp", ""),
                        "state_name": node_data.get("state_name", ""),
                    })
                
                active_items.sort(key=lambda x: x["priority"], reverse=True)
                
                result.working = {
                    "items": active_items,
                    "count": len(active_items),
                    "current": wm.get_current_state()
                }
            except Exception:
                pass
        
        return result
    
    def get(
        self,
        memory_type: str,
        limit: int = 10,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if memory_type == "semantic":
            from memory.semantic_memory import query_semantic_profile
            raw = query_semantic_profile(
                user_id=self.user_id,
                namespace=self.namespace,
                limit=limit
            )
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict):
                return raw.get("results", [])
            return []
        
        elif memory_type == "episodic":
            from memory.episodic_memory import get_recent_episodes
            raw = get_recent_episodes(limit=limit)
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict):
                return raw.get("results", [])
            return []
        
        elif memory_type == "procedural":
            from memory.procedural_memory import query_procedural_memory
            raw = query_procedural_memory("", limit=limit)
            if isinstance(raw, list):
                return raw
            elif isinstance(raw, dict):
                return raw.get("results", raw.get("rules", []))
            return []
        
        elif memory_type == "working":
            wm = self._get_working_memory()
            current = wm.get_current_state()
            return [current] if current else []
        
        else:
            raise ValueError(f"Unsupported memory types: {memory_type}")
    
    def get_context_for_llm(
        self,
        query: str,
        top_k: int = 3,
        max_length: int = 2000,
    ) -> str:
        results = self.search(query, top_k=top_k)
        context = results.to_context(max_items=top_k)

        if len(context) > max_length:
            context = context[:max_length - 3] + "..."
        
        return context
    
    def get_user_profile(
        self,
        categories: Optional[List[str]] = None,
        max_per_category: int = 5,
        format: str = "snapshot",  # "snapshot" | "dict"
    ) -> Union[str, Dict[str, List[Dict[str, Any]]]]:
        try:
            if format == "snapshot":
                from memory.semantic_memory import get_profile_snapshot
                return get_profile_snapshot(
                    user_id=self.user_id,
                    namespace=self.namespace,
                    max_per_relation=max_per_category
                )
            else:
                from memory.semantic_memory import get_full_profile
                return get_full_profile(
                    user_id=self.user_id,
                    namespace=self.namespace
                )
        except Exception:
            return "[User Profile]\n(Failed to retrieve)" if format == "snapshot" else {}

    def get_all_memories(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:

        return {
            "semantic": self.get("semantic", limit=limit),
            "episodic": self.get("episodic", limit=limit),
            "procedural": self.get("procedural", limit=limit),
            "working": self.get("working"),
        }
    
    def clear(self, memory_type: Optional[str] = None) -> bool:

        return True
    
    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "config": {
                "enable_semantic": self.config.enable_semantic,
                "enable_episodic": self.config.enable_episodic,
                "enable_procedural": self.config.enable_procedural,
                "enable_working": self.config.enable_working,
            },
        }

        try:
            stats["semantic_count"] = len(self.get("semantic", limit=1000))
        except Exception:
            stats["semantic_count"] = "N/A"
        
        try:
            stats["episodic_count"] = len(self.get("episodic", limit=1000))
        except Exception:
            stats["episodic_count"] = "N/A"
        
        try:
            stats["procedural_count"] = len(self.get("procedural", limit=1000))
        except Exception:
            stats["procedural_count"] = "N/A"
        
        return stats

    def extract_semantic_from_episodes(
        self,
        batch_size: int = 10,
        min_episodes: int = 5,
    ) -> Dict[str, Any]:
        from memory.episodic_memory import extract_to_semantic
        return extract_to_semantic(
            batch_size=batch_size,
            min_episodes=min_episodes,
            enable_consolidation=True,
        )
    
    def extract_procedural_from_episodes(
        self,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        from memory.episodic_memory import extract_to_procedural
        return extract_to_procedural(
            batch_size=batch_size,
            enable_deduplication=True,
        )
    
    def run_knowledge_evolution(self) -> Dict[str, Any]:
        from memory.knowledge_evolution import run_knowledge_evolution
        return run_knowledge_evolution()


__all__ = [
    "Memory",
    "MemoryConfig",
    "MemoryResult",
    "SearchResult",
]
