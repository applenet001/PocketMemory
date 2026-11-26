"""
Knowledge Evaluation
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
import re


@dataclass
class EvolutionPattern:
    pattern_type: str  # "preference", "habit", "knowledge", "procedure"
    content: str
    frequency: int
    first_seen: str
    last_seen: str
    source_episodes: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EvolutionResult:
    success: bool
    detected_patterns: List[EvolutionPattern] = field(default_factory=list)
    evolved_semantic: List[Dict[str, Any]] = field(default_factory=list)
    evolved_procedural: List[Dict[str, Any]] = field(default_factory=list)
    skipped_duplicates: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def run_knowledge_evolution(
    episodic_store=None,
    frequency_threshold: int = 3,
    time_window_days: int = 30,
    max_patterns: int = 10,
) -> EvolutionResult:
    result = EvolutionResult(success=False)

    try:
        if episodic_store is None:
            from memory.episodic_memory import _get_store
            episodic_store = _get_store()

        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_episodes = _get_recent_episodes(episodic_store, cutoff_date)
        
        if not recent_episodes:
            result.success = True
            return result
        
    except Exception as e:
        print(f"[KnowledgeEvolution] Failed to retrieve the plot: {e}")
        return result

    patterns = _detect_patterns(recent_episodes, frequency_threshold)
    result.detected_patterns = patterns[:max_patterns]
    
    if not patterns:
        result.success = True
        return result

    for pattern in patterns:
        if pattern.pattern_type in ["preference", "habit", "knowledge"]:
            evolved = _evolve_to_semantic(pattern)
            if evolved:
                result.evolved_semantic.append(evolved)
        elif pattern.pattern_type == "procedure":
            evolved = _evolve_to_procedural(pattern)
            if evolved:
                result.evolved_procedural.append(evolved)
    
    result.success = True
    return result


def _get_recent_episodes(store, cutoff_date: datetime) -> List[Dict[str, Any]]:
    try:
        if hasattr(store, 'get_all_episodes'):
            all_episodes = store.get_all_episodes()
        elif hasattr(store, 'episodes'):
            all_episodes = list(store.episodes.values())
        else:
            from memory.episodic_memory import get_recent_episodes
            return get_recent_episodes(limit=1000)
        
        recent = []
        for ep in all_episodes:
            if isinstance(ep, dict):
                ep_time = ep.get("timestamp", ep.get("when", ""))
            else:
                ep_time = getattr(ep, "timestamp", getattr(ep, "when", ""))
            
            try:
                if ep_time:
                    ep_dt = datetime.fromisoformat(ep_time.replace("Z", "+00:00"))
                    if ep_dt >= cutoff_date:
                        if isinstance(ep, dict):
                            recent.append(ep)
                        else:
                            recent.append(ep.__dict__ if hasattr(ep, '__dict__') else {"data": str(ep)})
            except (ValueError, AttributeError):
                recent.append(ep if isinstance(ep, dict) else {"data": str(ep)})
        
        return recent
    
    except Exception as e:
        print(f"[KnowledgeEvolution] Failed to retrieve the plot: {e}")
        return []


def _detect_patterns(episodes: List[Dict], threshold: int) -> List[EvolutionPattern]:
    patterns = []
    preference_patterns = _extract_preference_patterns(episodes)
    habit_patterns = _extract_habit_patterns(episodes)
    procedure_patterns = _extract_procedure_patterns(episodes)

    for content, count in preference_patterns.items():
        if count >= threshold:
            patterns.append(EvolutionPattern(
                pattern_type="preference",
                content=content,
                frequency=count,
                first_seen="",
                last_seen="",
                confidence=min(count / 10, 1.0),
            ))
    
    for content, count in habit_patterns.items():
        if count >= threshold:
            patterns.append(EvolutionPattern(
                pattern_type="habit",
                content=content,
                frequency=count,
                first_seen="",
                last_seen="",
                confidence=min(count / 10, 1.0),
            ))
    
    for content, count in procedure_patterns.items():
        if count >= threshold:
            patterns.append(EvolutionPattern(
                pattern_type="procedure",
                content=content,
                frequency=count,
                first_seen="",
                last_seen="",
                confidence=min(count / 10, 1.0),
            ))

    patterns.sort(key=lambda p: p.frequency, reverse=True)
    
    return patterns


def _extract_preference_patterns(episodes: List[Dict]) -> Counter:
    preferences = Counter()

    preference_keywords = [
        r"喜欢\s*(.+?)(?:[。，,]|$)",
        r"偏好\s*(.+?)(?:[。，,]|$)",
        r"prefer\s+(.+?)(?:\.|,|$)",
        r"like\s+(.+?)(?:\.|,|$)",
    ]
    
    for ep in episodes:
        content = ep.get("what", ep.get("content", ep.get("summary", "")))
        if not content:
            continue
        
        for pattern in preference_keywords:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 2:
                    preferences[f"user likes{match.strip()}"] += 1
    
    return preferences


def _extract_habit_patterns(episodes: List[Dict]) -> Counter:
    habits = Counter()

    habit_keywords = [
        r"每天\s*(.+?)(?:[。，,]|$)",
        r"经常\s*(.+?)(?:[。，,]|$)",
        r"通常\s*(.+?)(?:[。，,]|$)",
        r"always\s+(.+?)(?:\.|,|$)",
        r"usually\s+(.+?)(?:\.|,|$)",
    ]
    
    for ep in episodes:
        content = ep.get("what", ep.get("content", ep.get("summary", "")))
        if not content:
            continue
        
        for pattern in habit_keywords:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 2:
                    habits[f"user habit{match.strip()}"] += 1
    
    return habits


def _extract_procedure_patterns(episodes: List[Dict]) -> Counter:
    procedures = Counter()
    
    for ep in episodes:
        status = ep.get("status", ep.get("execution_status", ""))
        if status not in ["success", "completed", "成功"]:
            continue

        steps = ep.get("how", ep.get("steps", ""))
        if steps:
            step_key = str(steps)[:100]
            procedures[step_key] += 1
    
    return procedures


def _evolve_to_semantic(pattern: EvolutionPattern) -> Optional[Dict[str, Any]]:
    try:
        from memory.semantic_memory import store_semantic_profile

        content = pattern.content

        from memory.semantic_memory import query_semantic_profile
        existing = query_semantic_profile(
            user_id="default",
            namespace="default",
            limit=50,
        )
        
        for ex in existing:
            relation = ex.get("relation", "")
            entity = ex.get("entity", "")
            ex_content = f"{relation} {entity}".strip()
            if _is_similar(content, ex_content):
                print(f"[KnowledgeEvolution] Skip redundant semantic memory: {content[:50]}")
                return None

        result = store_semantic_profile(
            text=content,
            user_id="default",
            namespace="default",
            evidence_episode_ids=None,
            use_llm=True,
        )
        
        return {
            "content": content,
            "pattern_type": pattern.pattern_type,
            "frequency": pattern.frequency,
            "stored": result.get("success", True),
        }
    
    except Exception as e:
        print(f"[KnowledgeEvolution] Failed to evolve to semantic memory: {e}")
        return None


def _evolve_to_procedural(pattern: EvolutionPattern) -> Optional[Dict[str, Any]]:
    try:
        from memory.procedural_memory import store_procedural_memory

        content = f"Successfully executed task sequence: {pattern.content}"
        
        result = store_procedural_memory(
            text=content,
            meta={
                "source": "knowledge_evolution",
                "frequency": pattern.frequency,
                "confidence": pattern.confidence,
            }
        )
        
        return {
            "content": content,
            "frequency": pattern.frequency,
            "rules_count": len(result.get("rules", [])),
            "stored": result.get("success", True),
        }
    
    except Exception as e:
        print(f"[KnowledgeEvolution] Failed to evolve to procedural memory: {e}")
        return None


def _is_similar(text1: str, text2: str, threshold: float = 0.8) -> bool:
    if not text1 or not text2:
        return False
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return False
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union >= threshold


__all__ = [
    "run_knowledge_evolution",
    "EvolutionResult",
    "EvolutionPattern",
]
