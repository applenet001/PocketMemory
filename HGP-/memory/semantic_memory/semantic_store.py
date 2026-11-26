import os
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    GraphDatabase = None
    NEO4J_AVAILABLE = False

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        pass

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

logger = logging.getLogger(__name__)


class SemanticStore:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "semantic"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self._init_constraints()
        self._init_vector_index()
        self.llm_model = os.getenv("DEEPSEEK_LLM_MODEL", "deepseek-chat")
        self.llm_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.llm_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    def _init_constraints(self):
        with self.driver.session(database=self.database) as session:
            session.run("""
                CREATE CONSTRAINT user_unique IF NOT EXISTS
                FOR (u:User) REQUIRE (u.user_id, u.namespace) IS NODE KEY
            """)
            session.run("""
                CREATE CONSTRAINT entity_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE (e.name, e.namespace) IS NODE KEY
            """)
    
    def _init_vector_index(self):
        with self.driver.session(database=self.database) as session:
            try:
                session.run("""
                    CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS
                    FOR (e:Entity) ON e.embedding
                    OPTIONS {
                        indexConfig: {
                            `vector.dimensions`: 512,
                            `vector.similarity_function`: 'cosine'
                        }
                    }
                """)
            except Exception:
                pass
    
    def add_triplet(
        self,
        user_id: str,
        relation: str,
        entity: str,
        namespace: str = "default",
        confidence: float = 0.8,
        evidence_episode_ids: Optional[List[str]] = None,
        entity_type: Optional[str] = None,
        entity_embedding: Optional[List[float]] = None,
        enable_smart_merge: bool = True
    ) -> Dict[str, Any]:
        evidence_ids = evidence_episode_ids or []
        relation_type = relation.upper().replace(" ", "_")
        
        if entity_type is None:
            entity_type = self._infer_entity_type(entity, relation)
        existing_relations = self._search_user_relations(user_id, relation_type, namespace)
        if enable_smart_merge and existing_relations:
            decision = self._llm_decide_merge(
                user_id=user_id,
                new_relation=(user_id, relation_type, entity),
                existing_relations=existing_relations,
                namespace=namespace
            )
            
            action = decision.get("action", "add")
            reason = decision.get("reason", "LLM判断")
            
            if action == "skip":
                return {"action": "skip", "reason": reason}
            
            elif action == "update":
                old_target = decision.get("old_target")
                if old_target:
                    self._delete_relation(user_id, relation_type, old_target, namespace)
                self._add_relation_internal(
                    user_id, relation_type, entity, namespace,
                    confidence, evidence_ids, entity_type, entity_embedding
                )
                return {"action": "update", "reason": reason, "old_target": old_target}
            
            elif action == "delete_and_add":
                to_delete = decision.get("to_delete", [])
                for old_entity in to_delete:
                    self._delete_relation(user_id, relation_type, old_entity, namespace)
                self._add_relation_internal(
                    user_id, relation_type, entity, namespace,
                    confidence, evidence_ids, entity_type, entity_embedding
                )
                return {"action": "delete_and_add", "reason": reason, "deleted": to_delete}

        self._add_relation_internal(
            user_id, relation_type, entity, namespace,
            confidence, evidence_ids, entity_type, entity_embedding
        )
        return {"action": "add", "reason": "New Relationship"}
    
    def _add_relation_internal(
        self,
        user_id: str,
        relation_type: str,
        entity: str,
        namespace: str,
        confidence: float,
        evidence_ids: List[str],
        entity_type: str,
        entity_embedding: Optional[List[float]]
    ):
        with self.driver.session(database=self.database) as session:
            query = f"""
            MERGE (u:User {{user_id: $user_id, namespace: $namespace}})
            ON CREATE SET u.created_at = datetime()
            SET u.updated_at = datetime()
            
            MERGE (e:Entity {{name: $entity, namespace: $namespace}})
            ON CREATE SET e.type = $entity_type,
                          e.created_at = datetime(),
                          e.embedding = $embedding
            
            MERGE (u)-[r:{relation_type}]->(e)
            ON CREATE SET r.confidence = $confidence,
                          r.created_at = datetime(),
                          r.evidence_episode_ids = $evidence_ids
            ON MATCH SET r.last_seen_at = datetime(),
                         r.confidence = $confidence,
                         r.evidence_episode_ids = $evidence_ids
            """
            session.run(
                query,
                user_id=user_id,
                namespace=namespace,
                entity=entity,
                entity_type=entity_type,
                embedding=entity_embedding,
                confidence=confidence,
                evidence_ids=evidence_ids
            )
    
    def _delete_relation(self, user_id: str, relation_type: str, entity: str, namespace: str):
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (u:User {{user_id: $user_id, namespace: $namespace}})
                  -[r:{relation_type}]->(e:Entity {{name: $entity, namespace: $namespace}})
            DELETE r
            WITH e
            WHERE NOT EXISTS((e)-[]-())
            DELETE e
            """
            session.run(query, user_id=user_id, entity=entity, namespace=namespace)
    
    def _search_user_relations(
        self, 
        user_id: str, 
        relation_type: str, 
        namespace: str
    ) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            query = f"""
            MATCH (u:User {{user_id: $user_id, namespace: $namespace}})
                  -[r:{relation_type}]->(e:Entity)
            RETURN e.name AS entity, 
                   e.type AS entity_type,
                   r.confidence AS confidence,
                   r.evidence_episode_ids AS evidence_ids
            """
            result = session.run(query, user_id=user_id, namespace=namespace)
            
            relations = []
            for record in result:
                relations.append({
                    "entity": record["entity"],
                    "entity_type": record["entity_type"],
                    "confidence": record["confidence"],
                    "evidence_ids": record["evidence_ids"]
                })
            return relations
    
    def _infer_entity_type(self, entity: str, relation: str) -> str:
        relation_lower = relation.lower()
        if relation_lower in ["likes", "dislikes", "prefers", "enjoys"]:
            return "PREFERENCE"
        elif relation_lower in ["works_as", "profession"]:
            return "PROFESSION"
        elif relation_lower in ["lives_in", "located_in"]:
            return "LOCATION"
        elif relation_lower in ["has_skill", "good_at"]:
            return "SKILL"
        return "GENERAL"
    
    def _llm_decide_merge(
        self,
        user_id: str,
        new_relation: tuple,
        existing_relations: List[Dict],
        namespace: str
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.llm_api_key,
                base_url=self.llm_base_url
            )
            
            _, relation_type, new_entity = new_relation
            
            existing_str = "\n".join([
                f"- {user_id} -- {relation_type} --> {rel['entity']}"
                for rel in existing_relations
            ])
            
            prompt = f"""You are a knowledge graph merging expert. Determine whether the new relationship should be added, updated, deleted, or skipped.

**New Relationship:**
{user_id} -- {relation_type} --> {new_entity}

**Existing Relationships:**
{existing_str}

**Rules:**
1. UPDATE: The new relationship is a more specific version of the old one (e.g., engineer → senior engineer)
2. DELETE_AND_ADD: The new relationship contradicts the old one (e.g., lives in Beijing → lives in Shanghai)
3. ADD: The new relationship can coexist with the old one (e.g., multiple interests)
4. SKIP: The new relationship is a duplicate of the old one

**Output JSON:**
{{"action": "add|update|delete_and_add|skip", "reason": "理由", "to_delete": [], "old_target": ""}}
"""
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.replace("```", "").strip()
            
            return json.loads(content)
        except Exception:
            return {"action": "add", "reason": "LLM call failed; added by default"}
    
    def query_knowledge(
        self,
        user_id: str,
        namespace: str = "default",
        relation_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            if relation_type:
                query = f"""
                MATCH (u:User {{user_id: $user_id, namespace: $namespace}})
                      -[r:{relation_type}]->(e:Entity)
                RETURN u.user_id AS user_id, type(r) AS relation,
                       e.name AS entity, e.type AS entity_type,
                       r.confidence AS confidence, r.evidence_episode_ids AS evidence_ids,
                       r.created_at AS created_at, r.last_seen_at AS last_seen_at
                ORDER BY r.last_seen_at DESC
                LIMIT $limit
                """
            else:
                query = """
                MATCH (u:User {user_id: $user_id, namespace: $namespace})
                      -[r]->(e:Entity)
                RETURN u.user_id AS user_id, type(r) AS relation,
                       e.name AS entity, e.type AS entity_type,
                       r.confidence AS confidence, r.evidence_episode_ids AS evidence_ids,
                       r.created_at AS created_at, r.last_seen_at AS last_seen_at
                ORDER BY r.last_seen_at DESC
                LIMIT $limit
                """
            
            result = session.run(query, user_id=user_id, namespace=namespace, limit=limit)
            
            triplets = []
            for record in result:
                triplets.append({
                    "user_id": record["user_id"],
                    "relation": record["relation"],
                    "entity": record["entity"],
                    "entity_type": record["entity_type"],
                    "confidence": record["confidence"],
                    "evidence_ids": record["evidence_ids"],
                    "created_at": str(record["created_at"]) if record["created_at"] else None,
                    "last_seen_at": str(record["last_seen_at"]) if record["last_seen_at"] else None
                })
            return triplets
    
    def close(self):
        if self.driver:
            self.driver.close()

_store_instance: Optional[SemanticStore] = None


def _get_store() -> SemanticStore:
    global _store_instance
    if _store_instance is None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_SEMANTIC_DATABASE", "semantic")
        _store_instance = SemanticStore(uri, user, password, database)
    return _store_instance

def add_user_knowledge(
    user_id: str,
    relation: str,
    entity: str,
    namespace: str = "default",
    confidence: float = 0.8,
    evidence_episode_ids: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    store = _get_store()
    return store.add_triplet(
        user_id=user_id,
        relation=relation,
        entity=entity,
        namespace=namespace,
        confidence=confidence,
        evidence_episode_ids=evidence_episode_ids,
        **kwargs
    )


def query_user_knowledge(
    user_id: str,
    namespace: str = "default",
    relation_type: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    store = _get_store()
    return store.query_knowledge(user_id, namespace, relation_type, limit)
