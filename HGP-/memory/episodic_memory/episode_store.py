import os
import json
import uuid
import re
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from enum import Enum
from rank_bm25 import BM25Okapi

from dotenv import load_dotenv

from memory.core.llm_config import LLMConfig
from memory.core.llm_extractor import SchemaBasedExtractor
from memory.core.episodic_graph_builder import EpisodicGraphBuilder

BASE_DIR = Path(__file__).resolve().parents[2]
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

STORAGE_DIR = Path(__file__).parent.parent / "graph" / "episodic_memory"
STORAGE_PATH = STORAGE_DIR / "episodes.jsonl"

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_EPISODIC_DATABASE = os.getenv("NEO4J_EPISODIC_DATABASE", "episodic")

def normalize_timestamp(timestamp_str: Optional[str]) -> Optional[str]:

    if not timestamp_str:
        return None

    if hasattr(timestamp_str, 'iso_format'):
        return timestamp_str.iso_format()

    if isinstance(timestamp_str, datetime):
        return timestamp_str.isoformat()

    if not isinstance(timestamp_str, str):
        timestamp_str = str(timestamp_str)

    if timestamp_str.count('-') >= 2 and ('T' in timestamp_str or ' ' in timestamp_str):
        try:
            parsed = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return timestamp_str if 'T' in timestamp_str else parsed.isoformat()
        except Exception:
            pass

    import re

    time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)', timestamp_str, re.IGNORECASE)
    date_match = re.search(r'on\s+(\d{1,2})\s+([a-zA-Z]+),?\s+(\d{4})', timestamp_str, re.IGNORECASE)
    
    if time_match and date_match:
        try:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2))
            am_pm = time_match.group(3).lower()
            
            day = int(date_match.group(1))
            month_str = date_match.group(2)
            year = int(date_match.group(3))

            if am_pm == 'pm' and hour != 12:
                hour += 12
            elif am_pm == 'am' and hour == 12:
                hour = 0

            months = {
                'january': 1, 'february': 2, 'march': 3, 'april': 4,
                'may': 5, 'june': 6, 'july': 7, 'august': 8,
                'september': 9, 'october': 10, 'november': 11, 'december': 12
            }
            month = months.get(month_str.lower())
            if month is None:
                return None
            dt = datetime(year, month, day, hour, minute, 0)
            return dt.isoformat()
        except Exception:
            return None

    return None


class EventType(Enum):
    USER_MESSAGE = "user_message"       
    TASK_EXECUTION = "task_execution"  
    SYSTEM_EVENT = "system_event"       


class ExecutionStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    PENDING = "pending"
    UNKNOWN = "unknown"

@dataclass
class Episode:
    id: str = ""
    timestamp: Optional[str] = None  
    event_type: str = EventType.USER_MESSAGE.value
    content: str = ""
    actors: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    execution_status: str = ExecutionStatus.UNKNOWN.value
    action_sequence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: List[float] = field(default_factory=list)
    prev_episode_id: str = ""
    next_episode_id: str = ""
    extracted_semantic: bool = False
    extracted_procedural: bool = False
    
    def __post_init__(self):
        if not self.id:
            self.id = uuid.uuid4().hex

        if self.timestamp:
            normalized = normalize_timestamp(self.timestamp)
            if normalized:
                self.timestamp = normalized
            else:
                self.timestamp = datetime.now().isoformat()
        else:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        return cls(
            id=data.get("id", ""),
            timestamp=data.get("timestamp", ""),
            event_type=data.get("event_type", EventType.USER_MESSAGE.value),
            content=data.get("content", ""),
            actors=data.get("actors", []),
            context=data.get("context", {}),
            execution_status=data.get("execution_status", ExecutionStatus.UNKNOWN.value),
            action_sequence=data.get("action_sequence", []),
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding", []),
            prev_episode_id=data.get("prev_episode_id", ""),
            next_episode_id=data.get("next_episode_id", ""),
            extracted_semantic=data.get("extracted_semantic", False),
            extracted_procedural=data.get("extracted_procedural", False),
        )


class EmbeddingHelper:
    _model = None
    MODEL_NAME = "distiluse-base-multilingual-cased-v2"
    
    @classmethod
    def _get_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(cls.MODEL_NAME, device='cpu')
        return cls._model
    
    @classmethod
    def encode(cls, text: str) -> List[float]:
        model = cls._get_model()
        return model.encode(text, convert_to_numpy=True).tolist()
    
    @classmethod
    def encode_batch(cls, texts: List[str]) -> List[List[float]]:
        model = cls._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


def _cosine_sim(a: List[float], b: List[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

class Neo4jEpisodeGraph:
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self._driver = None
        self._connected = False
    
    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            self._driver.verify_connectivity()
            self._connected = True
        return self._driver
    
    def is_connected(self) -> bool:
        self._get_driver()
        return self._connected
    
    def _init_constraints(self):
        if not self._connected:
            return
           
        queries = [
            "CREATE CONSTRAINT episode_id_unique IF NOT EXISTS FOR (e:Episode) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT actor_ns_name_unique IF NOT EXISTS FOR (a:Actor) REQUIRE (a.namespace, a.name) IS UNIQUE",
            "CREATE INDEX episode_timestamp_idx IF NOT EXISTS FOR (e:Episode) ON (e.timestamp)",
            "CREATE INDEX episode_type_idx IF NOT EXISTS FOR (e:Episode) ON (e.event_type)",
            "CREATE INDEX episode_status_idx IF NOT EXISTS FOR (e:Episode) ON (e.execution_status)",
        ]
        
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            for query in queries:
                try:
                    session.run(query)
                except Exception:
                    pass
    
    def add_episode(
        self,
        episode: Episode,
        persons: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        extracted_location: Optional[str] = None,
        time_value: Optional[Dict] = None
    ) -> str:
        if not self.is_connected():
            return episode.id

        builder = EpisodicGraphBuilder(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
            database=NEO4J_EPISODIC_DATABASE
        )
        time_info = None
        if episode.timestamp:
            try:
                dt = datetime.fromisoformat(episode.timestamp.replace('Z', '+00:00'))
                time_info = {
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "hour": dt.hour,
                    "minute": dt.minute,
                }
            except Exception:
                pass
        actors = [{"name": actor, "role": "user" if actor == "user" else "other"} 
                  for actor in (episode.actors or ["user"])]

        locations = []
        if extracted_location:
            if isinstance(extracted_location, list):
                for loc in extracted_location:
                    if isinstance(loc, dict):
                        locations.append(loc)
                    else:
                        locations.append({
                            "name": str(loc),
                            "type": "extracted"
                        })
            else:
                locations.append({
                    "name": str(extracted_location),
                    "type": "extracted"
                })

        if episode.context.get("location"):
            locations.append({
                "name": episode.context["location"],
                "type": "context"
            })

        if persons is None:
            persons = []
        if topics is None:
            topics = []

        builder.build_episode(
            episode_id=episode.id,
            content=episode.content,
            timestamp=episode.timestamp,
            event_type=episode.event_type,
            execution_status=episode.execution_status,
            actors=actors,
            time_info=time_info,
            locations=locations,
            persons=persons,
            topics=topics,
            namespace=self.namespace,
            source_app=episode.metadata.get("source_app")
        )
        
        builder.close()
        return episode.id
    
    def link_episodes(self, prev_id: str, next_id: str, relation: str = "NEXT"):
        if not self.is_connected():
            return
        builder = EpisodicGraphBuilder(
            NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, 
            database=NEO4J_EPISODIC_DATABASE
        )
        
        builder.link_episodes(prev_id, next_id, namespace=self.namespace)
        builder.close()
    
    def query_timeline(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        
        conditions = ["e.namespace = $namespace"]  
        params = {"limit": limit, "namespace": self.namespace}
        
        if start_time:
            conditions.append("e.timestamp >= $start_time")
            params["start_time"] = start_time
        if end_time:
            conditions.append("e.timestamp <= $end_time")
            params["end_time"] = end_time
        if event_type:
            conditions.append("e.event_type = $event_type")
            params["event_type"] = event_type
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"""
        MATCH (e:Episode)
        {where_clause}
        OPTIONAL MATCH (e)-[:INVOLVES]->(a:Actor)
        RETURN e, collect(DISTINCT a.name) AS actors
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        results = []
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            records = session.run(query, params)
            for record in records:
                node = record["e"]
                results.append({
                    "id": node.get("episode_id") or node.get("id"),
                    "timestamp": node["timestamp"],
                    "event_type": node["event_type"],
                    "content": node["content"],
                    "execution_status": node["execution_status"],
                    "actors": record["actors"],
                })
        
        return results
    
    def query_by_actor(self, actor_name: str, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.is_connected():
            return []
        
        query = """
        MATCH (a:Actor {name: $actor_name, namespace: $namespace})-[:INVOLVES]-(e:Episode)
        WHERE e.namespace = $namespace
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        
        results = []
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            records = session.run(query, {
                "actor_name": actor_name,
                "limit": limit,
                "namespace": self.namespace
            })
            for record in records:
                node = record["e"]
                results.append(dict(node))
        
        return results
    
    def query_causal_chain(self, episode_id: str, direction: str = "forward", depth: int = 5) -> List[Dict]:
        if not self.is_connected():
            return []
        
        if direction == "forward":
            query = """
            MATCH path = (start:Episode {id: $id, namespace: $namespace})-[:NEXT|CAUSES*1..5]->(e:Episode)
            WHERE e.namespace = $namespace
            RETURN e
            ORDER BY e.timestamp
            LIMIT $depth
            """
        else:
            query = """
            MATCH path = (e:Episode)-[:NEXT|CAUSES*1..5]->(start:Episode {id: $id, namespace: $namespace})
            WHERE e.namespace = $namespace
            RETURN e
            ORDER BY e.timestamp DESC
            LIMIT $depth
            """
        
        results = []
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            records = session.run(query, {
                "id": episode_id,
                "depth": depth,
                "namespace": self.namespace
            })
            for record in records:
                results.append(dict(record["e"]))
        
        return results
    
    def get_unextracted_episodes(
        self,
        extract_type: str,  # "semantic" or "procedural"
        limit: int = 100
    ) -> List[Dict]:
        if not self.is_connected():
            return []
        
        if extract_type == "semantic":
            condition = "e.extracted_semantic = false OR e.extracted_semantic IS NULL"
        else:
            condition = "(e.extracted_procedural = false OR e.extracted_procedural IS NULL) AND e.execution_status = 'success'"
        
        query = f"""
        MATCH (e:Episode)
        WHERE {condition}
        RETURN e
        ORDER BY e.timestamp
        LIMIT $limit
        """
        
        results = []
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            records = session.run(query, {"limit": limit})
            for record in records:
                results.append(dict(record["e"]))
        
        return results
    
    def mark_extracted(self, episode_id: str, extract_type: str):
        if not self.is_connected():
            return
        
        field = "extracted_semantic" if extract_type == "semantic" else "extracted_procedural"
        query = f"""
        MATCH (e:Episode {{id: $id}})
        SET e.{field} = true
        """
        
        with self._driver.session(database=NEO4J_EPISODIC_DATABASE) as session:
            session.run(query, {"id": episode_id})
    
    def close(self):
        if self._driver:
            self._driver.close()

class EpisodeStore:
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        use_neo4j: bool = True,
        namespace: str = "default"
    ):

        self.namespace = namespace

        if storage_path is None:
            storage_dir = STORAGE_DIR / namespace
            storage_path = storage_dir / "episodes.jsonl"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._episode_cache: Dict[str, Episode] = {}
        self._cache_loaded = False

        self._use_neo4j = use_neo4j
        self._graph = Neo4jEpisodeGraph(namespace=namespace) if use_neo4j else None

        self._llm_extractor = None
        if use_neo4j:
            
            api_key, base_url, model = LLMConfig.get_llm_config()
            self._llm_extractor = SchemaBasedExtractor(
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
        self._last_episode_id: Optional[str] = None
    
    def _load_cache(self):
        if self._cache_loaded:
            return
        
        if self.storage_path.exists():
            with self.storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    episode = Episode.from_dict(data)
                    self._episode_cache[episode.id] = episode
                    self._last_episode_id = episode.id
        
        self._cache_loaded = True
    
    def add_episode(
        self,
        episode: Episode,
        link_to_previous: bool = True
    ) -> str:
        self._load_cache()
    
        if not episode.embedding:
            content_for_embed = episode.content
            
            if content_for_embed.startswith("【"):
                end_bracket = content_for_embed.find("】")
                if end_bracket > 0:
                    content_for_embed = content_for_embed[end_bracket + 1:]
            
            episode.embedding = EmbeddingHelper.encode(content_for_embed)
        
        if link_to_previous and self._last_episode_id:
            episode.prev_episode_id = self._last_episode_id
            if self._last_episode_id in self._episode_cache:
                self._episode_cache[self._last_episode_id].next_episode_id = episode.id

        self._episode_cache[episode.id] = episode
        self._last_episode_id = episode.id

        self._append_to_file(episode)

        persons = []
        topics = []
        extracted_location = None
        time_value = None

        if self._llm_extractor:
            try:
                extracted = self._llm_extractor.extract_from_episode(
                    episode_content=episode.content,
                    namespace=self.namespace
                )
                persons = extracted.get("persons", [])
                topics = extracted.get("topics", [])
                extracted_location = extracted.get("location")
                time_value = extracted.get("time")
            except Exception:
                pass

        if self._graph and self._graph.is_connected():

            self._graph.add_episode(
                episode,
                persons=persons,
                topics=topics,
                extracted_location=extracted_location,
                time_value=time_value
            )
            if episode.prev_episode_id:
                self._graph.link_episodes(episode.prev_episode_id, episode.id, "NEXT")       
        return episode.id
    
    def add_task_execution(
        self,
        task_description: str,
        action_sequence: List[str],
        success: bool,
        context: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ) -> str:

        episode = Episode(
            event_type=EventType.TASK_EXECUTION.value,
            content=f"Task: {task_description}\nStep: {' → '.join(action_sequence)}",
            action_sequence=action_sequence,
            execution_status=ExecutionStatus.SUCCESS.value if success else ExecutionStatus.FAILURE.value,
            context=context or {},
            metadata=metadata or {},
        )
        
        return self.add_episode(episode)
    
    def _append_to_file(self, episode: Episode):
        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(episode.to_dict(), ensure_ascii=False) + "\n")
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        use_graph_first: bool = True,
        graph_weight: float = 0.8,
        vector_weight: float = 0.2,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Episode, float]]:
        graph_results = []
        vector_results = []
        
        if use_graph_first and self._llm_extractor:
            
            extracted = self._llm_extractor.extract_from_episode(
                    episode_content=query,
                    namespace=self.namespace
                )
            participants = extracted.get("participants", [])
            persons = extracted.get("persons", [])
            topics = extracted.get("topics", [])
            time_info = extracted.get("time")
            time_text = time_info.get("value") if isinstance(time_info, dict) else time_info
            locations = extracted.get("location", [])
                
            has_participants = len(extracted.get("participants", [])) > 0
            has_time = isinstance(time_text, str) and bool(time_text.strip())
            has_location = len(extracted.get("location", [])) > 0
            has_persons = len(persons) > 0
            has_topics = len(topics) > 0
                
            if has_participants or has_time or has_location or has_persons or has_topics:
                    try:
                        from memory.core.episodic_graph_builder import EpisodicGraphBuilder
                        graph_builder = EpisodicGraphBuilder(
                            neo4j_uri=NEO4J_URI,
                            neo4j_user=NEO4J_USER,
                            neo4j_password=NEO4J_PASSWORD,
                            database=NEO4J_EPISODIC_DATABASE
                        )

                        episode_match_dims = {}  # ep_id -> {'actor': bool, 'person': bool, 'topic': bool, 'location': bool, 'time': bool}

                        if has_participants:
                            for participant_name in participants[:3]:
                                eps = graph_builder.query_episodes(
                                    namespace=self.namespace,
                                    actor_name=participant_name,
                                    limit=top_k,
                                    fuzzy_match=True,
                                    use_vector_search=True,  
                                    similarity_threshold=0.3  
                                )
                                for ep_dict in eps:
                                    ep_id = ep_dict.get('episode_id')
                                    if ep_id not in episode_match_dims:
                                        episode_match_dims[ep_id] = {'actor': False, 'person': False, 'topic': False, 'location': False, 'time': False}
                                    episode_match_dims[ep_id]['actor'] = True

                        if has_location:
                            for location_name in locations[:3]:
                                eps = graph_builder.query_episodes(
                                    namespace=self.namespace,
                                    location_name=location_name,
                                    limit=top_k,
                                    fuzzy_match=True,
                                    use_vector_search=True,  
                                    similarity_threshold=0.3  
                                )
                                for ep_dict in eps:
                                    ep_id = ep_dict.get('episode_id')
                                    if ep_id not in episode_match_dims:
                                        episode_match_dims[ep_id] = {'actor': False, 'person': False, 'topic': False, 'location': False, 'time': False}
                                    episode_match_dims[ep_id]['location'] = True

                        if has_time:
                            eps = graph_builder.query_episodes(
                                namespace=self.namespace,
                                event_date=time_text,
                                limit=top_k * 2,
                            )
                            for ep_dict in eps:
                                ep_id = ep_dict.get('episode_id')
                                if ep_id not in episode_match_dims:
                                    episode_match_dims[ep_id] = {'actor': False, 'person': False, 'topic': False, 'location': False, 'time': False}
                                episode_match_dims[ep_id]['time'] = True

                        if has_persons:
                            for person_name in persons[:3]:
                                eps = graph_builder.query_episodes(
                                    namespace=self.namespace,
                                    person_name=person_name,
                                    limit=top_k,
                                    fuzzy_match=True,
                                    use_vector_search=True,  
                                    similarity_threshold=0.3  
                                )
                                for ep_dict in eps:
                                    ep_id = ep_dict.get('episode_id')
                                    if ep_id not in episode_match_dims:
                                        episode_match_dims[ep_id] = {'actor': False, 'person': False, 'topic': False, 'location': False, 'time': False}
                                    episode_match_dims[ep_id]['person'] = True

                        if has_topics:
                            for topic_name in topics[:3]:
                                eps = graph_builder.query_episodes(
                                    namespace=self.namespace,
                                    topic_name=topic_name,
                                    limit=top_k,
                                    fuzzy_match=True,
                                    use_vector_search=True,  
                                    similarity_threshold=0.3  
                                )
                                for ep_dict in eps:
                                    ep_id = ep_dict.get('episode_id')
                                    if ep_id not in episode_match_dims:
                                        episode_match_dims[ep_id] = {'actor': False, 'person': False, 'topic': False, 'location': False, 'time': False}
                                    episode_match_dims[ep_id]['topic'] = True
                        
                        seen_ids = set()
                        for ep_id, match_dims in episode_match_dims.items():
                            if ep_id in self._episode_cache and ep_id not in seen_ids:
                                ep = self._episode_cache[ep_id]

                                if metadata_filter and not self._match_metadata_filter(ep, metadata_filter):
                                    continue

                                matched_dims = sum(1 for v in match_dims.values() if v)
                                graph_score = min(0.25 + matched_dims * 0.15, 1.0)  
                                
                                graph_results.append((ep, graph_score))
                                seen_ids.add(ep_id)
                        
                        graph_builder.close()
                        
                    except Exception:
                        pass
                        
        
        vector_candidate_count = top_k * 2
        try:
            vector_results = self.vector_search(
                query, 
                top_k=vector_candidate_count,
                threshold=threshold,
                metadata_filter=metadata_filter
            )
        except Exception:
            pass
        
        fusion_dict = {}
        
        for ep, graph_score in graph_results:
            if ep.id not in fusion_dict:
                fusion_dict[ep.id] = {
                    'episode': ep,
                    'graph_score': graph_score,
                    'vector_score': 0.0,
                    'source': 'graph'
                }
        
        for ep, vector_score in vector_results:
            if ep.id not in fusion_dict:
                fusion_dict[ep.id] = {
                    'episode': ep,
                    'graph_score': 0.0,
                    'vector_score': vector_score,
                    'source': 'vector'
                }
            else:
                fusion_dict[ep.id]['vector_score'] = vector_score
                fusion_dict[ep.id]['source'] = 'both'

        final_results = []
        for ep_id, scores in fusion_dict.items():
            ep = scores['episode']
            g_score = scores['graph_score']
            v_score = scores['vector_score']
            source = scores['source']
            
            if source == 'both':
                final_score = g_score * graph_weight + v_score * vector_weight + 0.1
            elif source == 'graph':
                final_score = g_score
            elif source == 'vector':
                final_score = min(v_score*1.2, 1)            
            final_results.append((ep, final_score))

        final_results.sort(key=lambda x: x[1], reverse=True)

        final_results = final_results[:top_k]
         
        final_results = self._bm25_rerank(query, final_results, top_k)
        
        if len(final_results) == 0:
            return []
        
        return final_results
    
    def _bm25_rerank(
        self,
        query: str,
        candidates: List[Tuple[Episode, float]],
        top_k: int
    ) -> List[Tuple[Episode, float]]:
        if len(candidates) == 0:
            return []

        documents = []
        for ep, _ in candidates:
            doc_text = ep.content
            tokens = self._tokenize(doc_text)
            documents.append(tokens)

        bm25 = BM25Okapi(documents)

        query_tokens = self._tokenize(query)
   
        bm25_scores = bm25.get_scores(query_tokens)
        
        reranked = []
        for i, (ep, orig_score) in enumerate(candidates):
            bm25_score = bm25_scores[i]
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
            normalized_bm25 = bm25_score / max_bm25
            
            has_graph_match = orig_score >= 0.4
            
            if has_graph_match:
                vector_score_inferred = orig_score  
                combined_score = orig_score * 0.5 + vector_score_inferred * 0.35 + normalized_bm25 * 0.15
                fusion_type = "Graph+Vector+BM25"
            else:
                combined_score = orig_score * 0.6 + normalized_bm25 * 0.4
                fusion_type = "Vector+BM25"
            
            reranked.append((ep, combined_score))

        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked[:top_k]
    
    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', text.lower())
        
        tokens = []
        for word in text.split():
            if re.search(r'[\u4e00-\u9fa5]', word):
                tokens.extend(list(word))
            elif len(word) > 1:
                tokens.append(word)
        
        return tokens
    
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Episode, float]]: 
        self._load_cache()
        
        if not self._episode_cache:
            return []
        
        query_emb = EmbeddingHelper.encode(query)
        results = []
        
        for episode in self._episode_cache.values():
            if metadata_filter and not self._match_metadata_filter(episode, metadata_filter):
                continue
            
            if episode.embedding:
                sim = _cosine_sim(query_emb, episode.embedding)
                if sim >= threshold:
                    results.append((episode, sim))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _match_metadata_filter(self, episode: Episode, filter_dict: Dict[str, Any]) -> bool:
        metadata = episode.metadata or {}
        
        for field, condition in filter_dict.items():
            value = metadata.get(field)
            
            if isinstance(condition, dict):
                for op, target in condition.items():
                    if not self._check_operator(value, op, target):
                        return False
            else:
                if value != condition:
                    return False
        
        return True
    
    def _check_operator(self, value: Any, op: str, target: Any) -> bool:
        if value is None:
            return False
        
        try:
            if op == "$lt":
                return value < target
            elif op == "$lte":
                return value <= target
            elif op == "$gt":
                return value > target
            elif op == "$gte":
                return value >= target
            elif op == "$eq":
                return value == target
            elif op == "$ne":
                return value != target
            elif op == "$in":
                return value in target
            else:
                return False
        except Exception:
            return False
    
    def query_recent_episodes(self, limit: int = 10) -> List[Dict[str, Any]]:
        if self._graph and self._graph.is_connected():
            results = self._graph.query_timeline(
                start_time=None,
                end_time=None,
                event_type=None,
                limit=limit
            )
            return results
        self._load_cache()
        episodes = list(self._episode_cache.values())
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in episodes[:limit]]
    
    def query_timeline(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Episode]:
        if self._graph and self._graph.is_connected():
            results = self._graph.query_timeline(start_time, end_time, event_type, limit)
            return [Episode.from_dict(r) for r in results]

        self._load_cache()
        episodes = list(self._episode_cache.values())
        
        if start_time:
            episodes = [e for e in episodes if e.timestamp >= start_time]
        if end_time:
            episodes = [e for e in episodes if e.timestamp <= end_time]
        if event_type:
            episodes = [e for e in episodes if e.event_type == event_type]
        
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]
    
    def query_by_actor(self, actor_name: str, limit: int = 50) -> List[Episode]:
        if self._graph and self._graph.is_connected():
            results = self._graph.query_by_actor(actor_name, limit)
            return [Episode.from_dict(r) for r in results]
        
        self._load_cache()
        episodes = [e for e in self._episode_cache.values() if actor_name in e.actors]
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]
    
    def query_causal_chain(
        self,
        episode_id: str,
        direction: str = "forward",
        depth: int = 5
    ) -> List[Episode]:
        if self._graph and self._graph.is_connected():
            results = self._graph.query_causal_chain(episode_id, direction, depth)
            return [Episode.from_dict(r) for r in results]
        
        self._load_cache()
        chain = []
        current_id = episode_id
        
        for _ in range(depth):
            if current_id not in self._episode_cache:
                break
            episode = self._episode_cache[current_id]
            chain.append(episode)
            current_id = episode.next_episode_id if direction == "forward" else episode.prev_episode_id
            if not current_id:
                break
        
        return chain
    
    def get_successful_task_sequences(self, limit: int = 100) -> List[Episode]:
        self._load_cache()
        
        episodes = [
            e for e in self._episode_cache.values()
            if e.event_type == EventType.TASK_EXECUTION.value
            and e.execution_status == ExecutionStatus.SUCCESS.value
            and e.action_sequence
            and not e.extracted_procedural
        ]
        
        episodes.sort(key=lambda e: e.timestamp, reverse=True)
        return episodes[:limit]
    
    def get_unextracted_for_semantic(self, limit: int = 100) -> List[Episode]:
        self._load_cache()
        
        episodes = [
            e for e in self._episode_cache.values()
            if not e.extracted_semantic
        ]
        
        episodes.sort(key=lambda e: e.timestamp)
        return episodes[:limit]
    
    def mark_extracted(self, episode_id: str, extract_type: str):
        if episode_id in self._episode_cache:
            if extract_type == "semantic":
                self._episode_cache[episode_id].extracted_semantic = True
            else:
                self._episode_cache[episode_id].extracted_procedural = True
        
        if self._graph:
            self._graph.mark_extracted(episode_id, extract_type)
    
    def get_episode(self, episode_id: str) -> Optional[Episode]:
        self._load_cache()
        return self._episode_cache.get(episode_id)
    
    def get_stats(self) -> Dict[str, Any]:
        self._load_cache()
        
        type_counts = {}
        status_counts = {}
        
        for e in self._episode_cache.values():
            type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            status_counts[e.execution_status] = status_counts.get(e.execution_status, 0) + 1
        
        return {
            "total_episodes": len(self._episode_cache),
            "by_type": type_counts,
            "by_status": status_counts,
            "unextracted_semantic": sum(1 for e in self._episode_cache.values() if not e.extracted_semantic),
            "unextracted_procedural": sum(
                1 for e in self._episode_cache.values()
                if not e.extracted_procedural
                and e.execution_status == ExecutionStatus.SUCCESS.value
                and e.action_sequence
            ),
        }
