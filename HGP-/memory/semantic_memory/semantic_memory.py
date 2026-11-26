#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

from .semantic_store import SemanticStore, _get_store, add_user_knowledge, query_user_knowledge

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


class SemanticExtractor:
    
    def __init__(self):
        self.llm_model = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.store = _get_store()
    
    def extract_from_episode(
        self,
        episode_content: str,
        episode_id: str,
        user_id: str,
        namespace: str = "default",
        enable_smart_merge: bool = True
    ) -> List[Dict[str, Any]]:
        try:
            triplets = self._llm_extract_triplets(episode_content, user_id)
            
            if not triplets:
                return []

            results = []
            for triplet in triplets:
                relation = triplet.get("relation", "")
                entity = triplet.get("entity", "")
                entity_type = triplet.get("entity_type")
                confidence = triplet.get("confidence", 0.8)
                
                if not relation or not entity:
                    continue

                from memory.core.episodic_graph_builder import EntityEmbedder
                entity_embedding = EntityEmbedder.encode(entity)
                
                result = self.store.add_triplet(
                    user_id=user_id,
                    relation=relation,
                    entity=entity,
                    namespace=namespace,
                    confidence=confidence,
                    evidence_episode_ids=[episode_id],
                    entity_type=entity_type,
                    entity_embedding=entity_embedding,
                    enable_smart_merge=enable_smart_merge
                )
                
                results.append({
                    "triplet": triplet,
                    "action": result.get("action"),
                    "reason": result.get("reason")
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []
    
    def _llm_extract_triplets(
        self,
        content: str,
        user_id: str
    ) -> List[Dict[str, Any]]:
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url
            )
            
            prompt = f"""You are a knowledge graph expert responsible for extracting triples related to users' semantic memories.
            Semantic memory is defined as follows: Semantic memory stores long-term, context-free knowledge abstracted from individual experiences, which is used to describe user profiles (such as relationships, preferences, skills, and basic information). It can be reused across time and tasks.
            Extract knowledge triples about user “{user_id}” from the text below.

Text:
{content}

Extraction Rules:
1. Triplet format: (User ID, Relation, Entity)
2. Extract only profile information related to user {user_id}
3. Use standard verbs that describe user profiles for the “Relation” field, such as: likes, dislikes, works_as, lives_in, has_skill, good_at, studies, has, is (or other terms that meet the semantic memory definition for relations)
4. Entity refers to a specific object

Confidence Criteria (Examples):
0.90–1.00: Clear, direct, and unconditional statements in the text (first-person preferences/facts)
Examples: “I like basketball,” “I live in Shanghai,” “I am a doctor”
0.70–0.89: Clear evidence exists, but there is slight uncertainty (paraphrasing, omitted context, or synonymous expressions)
Examples: “I’ve been playing basketball a lot lately” (implies “likes,” but less definitive than stating it directly)
0.50–0.69: Weak evidence/inference (indirect clues, possibly a one-time event)
Examples: “I went to play basketball yesterday” (more like an episodic event, not necessarily a stable profile)
< 0.50: Conjecture, conditional statements, counterfactuals, sarcasm, or insufficient evidence; avoid outputting this triplet whenever possible (or output a low score)

Special notes:
Explicit negations (“I don’t like X”) should not produce “likes”; they should produce “dislikes” and be scored according to the same conventions.
Conditional preferences (“If I have time, I would…”) should be scored lower or not outputted.

**Example JSON array output:**
[{{“relation”: “likes”, “entity”: “basketball”, “entity_type”: ‘HOBBY’, “confidence”: 0.9}}]

Output only the JSON array; no additional text. If no triples are found, return []
"""
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content_text = response.choices[0].message.content.strip()
            
            if "```json" in content_text:
                content_text = content_text.split("```json")[1].split("```")[0].strip()
            elif "```" in content_text:
                content_text = content_text.replace("```", "").strip()
            
            triplets = json.loads(content_text)
            
            if not isinstance(triplets, list):
                return []
            
            return triplets
            
        except Exception as e:
            logger.error(f"LLM triplet extraction failed: {e}")
            return []
    
    def extract_from_episodes_batch(
        self,
        episodes: List[Dict[str, Any]],
        user_id: str,
        namespace: str = "default",
        enable_smart_merge: bool = True
    ) -> Dict[str, Any]:
        """Batch Extract Trios"""
        total_triplets = 0
        total_episodes = len(episodes)
        
        for episode in episodes:
            episode_id = episode.get("episode_id", "")
            content = episode.get("content", "")
            
            if not content:
                continue
            
            results = self.extract_from_episode(
                content, episode_id, user_id, namespace, enable_smart_merge
            )
            total_triplets += len(results)
        
        return {
            "total_episodes": total_episodes,
            "total_triplets": total_triplets,
            "avg_triplets_per_episode": total_triplets / total_episodes if total_episodes > 0 else 0
        }


_extractor_instance: Optional[SemanticExtractor] = None


def get_extractor() -> SemanticExtractor:
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = SemanticExtractor()
    return _extractor_instance

def store_semantic_profile(
    text: str,
    user_id: str,
    namespace: str = "default",
    evidence_episode_ids: Optional[List[str]] = None,
    use_llm: bool = True,
    enable_smart_merge: bool = True,
) -> Dict[str, Any]:
    if not use_llm:
        return {
            "success": True,
            "triplets_added": 0,
            "triplets_updated": 0,
            "triplets_skipped": 0,
            "mode": "no_llm"
        }
    
    episode_id = evidence_episode_ids[0] if evidence_episode_ids else f"manual_{user_id}"
    
    extractor = get_extractor()
    results = extractor.extract_from_episode(
        episode_content=text,
        episode_id=episode_id,
        user_id=user_id,
        namespace=namespace,
        enable_smart_merge=enable_smart_merge
    )
    
    added = sum(1 for r in results if r["action"] == "add")
    updated = sum(1 for r in results if r["action"] in ["update", "delete_and_add"])
    skipped = sum(1 for r in results if r["action"] == "skip")
    
    return {
        "success": True,
        "triplets_added": added,
        "triplets_updated": updated,
        "triplets_skipped": skipped,
        "total": len(results),
        "details": results
    }


def query_semantic_profile(
    user_id: str,
    namespace: str = "default",
    relation_type: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    return query_user_knowledge(
        user_id=user_id,
        namespace=namespace,
        relation_type=relation_type,
        limit=limit
    )


def extract_profile_from_episodes(
    episodes: List[Dict[str, Any]],
    user_id: str,
    namespace: str = "default",
    enable_smart_merge: bool = True
) -> Dict[str, Any]:
    extractor = get_extractor()
    return extractor.extract_from_episodes_batch(
        episodes=episodes,
        user_id=user_id,
        namespace=namespace,
        enable_smart_merge=enable_smart_merge
    )


def filter_by_vector_similarity(
    triplets: List[Dict[str, Any]],
    query: str,
    top_k: int = 10,
    threshold: float = 0.3
) -> List[Dict[str, Any]]:
    if not triplets or not query:
        return triplets[:top_k]
    
    try:
        import numpy as np
        from memory.core.episodic_graph_builder import EntityEmbedder
        
        query_embedding = EntityEmbedder.encode(query)
        
        scored_triplets = []
        for triplet in triplets:
            relation = triplet.get('relation', '')
            entity = triplet.get('entity', '')
            triplet_text = f"{relation} {entity}"
            
            triplet_embedding = EntityEmbedder.encode(triplet_text)
            similarity = np.dot(query_embedding, triplet_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(triplet_embedding) + 1e-8
            )
            
            if similarity >= threshold:
                triplet_copy = triplet.copy()
                triplet_copy['similarity'] = float(similarity)
                scored_triplets.append(triplet_copy)
        
        scored_triplets.sort(key=lambda x: x['similarity'], reverse=True)
        return scored_triplets[:top_k]
    
    except Exception:
        return triplets[:top_k]


def get_full_profile(
    user_id: str,
    namespace: str = "default",
    limit: int = 100
) -> Dict[str, List[Dict[str, Any]]]:
    triplets = query_user_knowledge(user_id, namespace, limit=limit)
    
    profile = {}
    for triplet in triplets:
        relation = triplet.get('relation', 'OTHER')
        if relation not in profile:
            profile[relation] = []
        profile[relation].append(triplet)
    
    return profile


def get_profile_snapshot(
    user_id: str,
    namespace: str = "default",
    max_per_relation: int = 5
) -> str:

    profile = get_full_profile(user_id, namespace)
    
    if not profile:
        return f"User {user_id} has no semantic profile records"
    
    lines = [f"【User Profile: {user_id}】"]
    
    # 关系类型中文映射
    relation_names = {
        "LIKES": "喜欢",
        "DISLIKES": "不喜欢",
        "WORKS_AS": "职业",
        "LIVES_IN": "居住地",
        "HAS_SKILL": "技能",
        "GOOD_AT": "擅长",
        "STUDIES": "学习",
        "HAS": "拥有",
        "IS": "是"
    }
    
    for relation, triplets in profile.items():
        display_name = relation_names.get(relation, relation)
        entities = [t['entity'] for t in triplets[:max_per_relation]]
        lines.append(f"- {display_name}: {', '.join(entities)}")
    
    return "\n".join(lines)
