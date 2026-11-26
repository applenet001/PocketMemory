import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class EntityEmbedder:
    _model = None
    MODEL_NAME = "distiluse-base-multilingual-cased-v2"
    EMBEDDING_DIM = 512  
    
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
        if not texts:
            return []
        model = cls._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]


class EpisodicGraphBuilder:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, database: str = "episodic"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.database = database
        self._init_constraints()
    
    def _init_constraints(self):
        try:
            with self.driver.session(database=self.database) as session:
                session.run("""
                    CREATE CONSTRAINT episode_namespace_unique IF NOT EXISTS
                    FOR (e:Episode) REQUIRE (e.episode_id, e.namespace) IS NODE KEY
                """)

                session.run("""
                    CREATE CONSTRAINT actor_namespace_unique IF NOT EXISTS
                    FOR (a:Actor) REQUIRE (a.name, a.namespace) IS NODE KEY
                """)

                session.run("""
                    CREATE CONSTRAINT time_namespace_unique IF NOT EXISTS
                    FOR (t:Time) REQUIRE (t.timestamp, t.namespace) IS NODE KEY
                """)

                session.run("""
                    CREATE CONSTRAINT location_namespace_unique IF NOT EXISTS
                    FOR (l:Location) REQUIRE (l.name, l.namespace) IS NODE KEY
                """)

                session.run("""
                    CREATE CONSTRAINT person_namespace_unique IF NOT EXISTS
                    FOR (p:Person) REQUIRE (p.name, p.namespace) IS NODE KEY
                """)

                session.run("""
                    CREATE CONSTRAINT topic_namespace_unique IF NOT EXISTS
                    FOR (t:Topic) REQUIRE (t.name, t.namespace) IS NODE KEY
                """)

                vector_indexes = [
                    ("person_embedding_index", "Person", "p"),
                    ("actor_embedding_index", "Actor", "a"),
                    ("topic_embedding_index", "Topic", "t"),
                    ("episode_embedding_index", "Episode", "e"),
                    ("location_embedding_index", "Location", "l"),
                ]
                for index_name, label, alias in vector_indexes:
                    try:
                        session.run(f"""
                            CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                            FOR ({alias}:{label})
                            ON ({alias}.embedding)
                            OPTIONS {{indexConfig: {{
                                `vector.dimensions`: 512,
                                `vector.similarity_function`: 'cosine'
                            }}}}
                        """)
                    except Exception:
                        pass
        except Exception:
            pass
    
    def build_episode(
        self,
        episode_id: str,
        content: str,
        timestamp: str,
        event_type: str = "user_message",
        execution_status: str = "unknown",
        actors: Optional[List[Dict[str, str]]] = None,
        time_info: Optional[Dict[str, Any]] = None,
        locations: Optional[List[Dict[str, str]]] = None,
        persons: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> bool:
        try:
            with self.driver.session(database=self.database) as session:

                episode_embedding = EntityEmbedder.encode(content)
                
                session.run(
                    """
                    MERGE (e:Episode {episode_id: $episode_id, namespace: $namespace})
                    SET e.content = $content,
                        e.timestamp = $timestamp,
                        e.event_type = $event_type,
                        e.execution_status = $execution_status,
                        e.embedding = $embedding,
                        e.updated_at = datetime()
                    """,
                    episode_id=episode_id,
                    content=content,
                    timestamp=timestamp,
                    event_type=event_type,
                    execution_status=execution_status,
                    embedding=episode_embedding,
                    namespace=namespace
                )

                if actors:
                    actor_names = [actor.get("name", "unknown") for actor in actors]
                    actor_embeddings = EntityEmbedder.encode_batch(actor_names)
                    
                    for actor, embedding in zip(actors, actor_embeddings):
                        actor_name = actor.get("name", "unknown")
                        actor_role = actor.get("role", "other")
                        
                        session.run(
                            """
                            MERGE (a:Actor {name: $name, namespace: $namespace})
                            ON CREATE SET a.role = $role,
                                          a.embedding = $embedding
                            ON MATCH SET a.embedding = COALESCE(a.embedding, $embedding)
                            
                            WITH a
                            MATCH (e:Episode {episode_id: $episode_id, namespace: $namespace})
                            MERGE (e)-[:INVOLVES]->(a)
                            """,
                            name=actor_name,
                            role=actor_role,
                            embedding=embedding,
                            episode_id=episode_id,
                            namespace=namespace
                        )

                if time_info:
                    year = time_info.get("year")
                    month = time_info.get("month", 1)
                    day = time_info.get("day", 1)

                    try:
                        year = int(year) if year else None
                        month = int(month) if month else 1
                        day = int(day) if day else 1
                    except (TypeError, ValueError):
                        year = None
                        month = 1
                        day = 1

                    if year or time_info.get("value") or time_info.get("relative"):
                        if year:
                            time_str = f"{year}-{month:02d}-{day:02d}"
                        else:
                            time_str = time_info.get("value") or time_info.get("relative") or "unknown"
                    
                    session.run(
                        """
                        MERGE (t:Time {timestamp: $timestamp, namespace: $namespace})
                        SET t.value = $value,
                            t.year = $year,
                            t.month = $month,
                            t.day = $day,
                            t.granularity = $granularity
                        
                        WITH t
                        MATCH (e:Episode {episode_id: $episode_id, namespace: $namespace})
                        MERGE (e)-[:HAPPENS_AT]->(t)
                        """,
                        timestamp=time_str,
                        value=time_info.get("value", ""),
                        year=year,
                        month=month,
                        day=day,
                        granularity=time_info.get("granularity", "day"),
                        episode_id=episode_id,
                        namespace=namespace
                    )

                if locations:
                    location_names = [loc.get("name", "unknown") for loc in locations]
                    location_embeddings = EntityEmbedder.encode_batch(location_names)
                    
                    for loc, embedding in zip(locations, location_embeddings):
                        loc_name = loc.get("name", "unknown")
                        loc_type = loc.get("type", "physical")
                        
                        session.run(
                            """
                            MERGE (l:Location {name: $name, namespace: $namespace})
                            ON CREATE SET l.type = $type,
                                          l.embedding = $embedding
                            ON MATCH SET l.embedding = COALESCE(l.embedding, $embedding)
                            
                            WITH l
                            MATCH (e:Episode {episode_id: $episode_id, namespace: $namespace})
                            MERGE (e)-[:OCCURS_AT]->(l)
                            """,
                            name=loc_name,
                            type=loc_type,
                            embedding=embedding,
                            episode_id=episode_id,
                            namespace=namespace
                        )
                if persons:
                    person_embeddings = EntityEmbedder.encode_batch(persons)
                    for person_name, embedding in zip(persons, person_embeddings):
                        session.run(
                            """
                            MERGE (p:Person {name: $name, namespace: $namespace})
                            ON CREATE SET p.embedding = $embedding
                            ON MATCH SET p.embedding = COALESCE(p.embedding, $embedding)
                            
                            WITH p
                            MATCH (e:Episode {episode_id: $episode_id, namespace: $namespace})
                            MERGE (e)-[:MENTIONS]->(p)
                            """,
                            name=person_name,
                            embedding=embedding,
                            episode_id=episode_id,
                            namespace=namespace
                        )
                if topics:
                    topic_embeddings = EntityEmbedder.encode_batch(topics)
                    for topic_name, embedding in zip(topics, topic_embeddings):
                        session.run(
                            """
                            MERGE (t:Topic {name: $name, namespace: $namespace})
                            ON CREATE SET t.embedding = $embedding
                            ON MATCH SET t.embedding = COALESCE(t.embedding, $embedding)
                            
                            WITH t
                            MATCH (e:Episode {episode_id: $episode_id, namespace: $namespace})
                            MERGE (e)-[:ABOUT]->(t)
                            """,
                            name=topic_name,
                            embedding=embedding,
                            episode_id=episode_id,
                            namespace=namespace
                        )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to construct the plot graph：{e}")
            return False
    
    def link_episodes(
        self,
        prev_episode_id: str,
        next_episode_id: str,
        namespace: str = "default"
    ) -> bool:
        try:
            with self.driver.session(database=self.database) as session:
                session.run(
                    """
                    MATCH (e1:Episode {episode_id: $prev_id, namespace: $namespace})
                    MATCH (e2:Episode {episode_id: $next_id, namespace: $namespace})
                    MERGE (e1)-[:NEXT]->(e2)
                    """,
                    prev_id=prev_episode_id,
                    next_id=next_episode_id,
                    namespace=namespace
                )
                return True
        except Exception as e:
            logger.error(f"Failed to link episodes: {e}")
            return False
    
    def query_episodes(
        self,
        namespace: str = "default",
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        actor_name: Optional[str] = None,
        location_name: Optional[str] = None,
        person_name: Optional[str] = None,
        topic_name: Optional[str] = None,
        limit: int = 10,
        fuzzy_match: bool = True,
        use_vector_search: bool = True,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        try:
            results = []
            
            with self.driver.session(database=self.database) as session:
                if actor_name and use_vector_search:
                    results.extend(self._vector_search_actor(
                        session, namespace, actor_name, limit, similarity_threshold
                    ))
                
                if location_name and use_vector_search:
                    results.extend(self._vector_search_location(
                        session, namespace, location_name, limit, similarity_threshold
                    ))
                
                if person_name and use_vector_search:
                    results.extend(self._vector_search_person(
                        session, namespace, person_name, limit, similarity_threshold
                    ))
                
                if topic_name and use_vector_search:
                    results.extend(self._vector_search_topic(
                        session, namespace, topic_name, limit, similarity_threshold
                    ))
                
                if start_time or end_time:
                    results.extend(self._time_range_query(
                        session, namespace, start_time, end_time, limit
                    ))
                
                if not results and fuzzy_match:
                    fallback_results = self._structured_query(
                        session, namespace, actor_name, location_name, limit
                    )
                    results.extend(fallback_results)

                    if not results and (person_name or topic_name):
                        fallback_results = self._fallback_string_match(
                            session, namespace, person_name, topic_name, limit
                        )
                        results.extend(fallback_results)
                seen_ids = set()
                unique_results = []
                for ep in sorted(results, key=lambda x: x.get('similarity', 0), reverse=True):
                    if ep['episode_id'] not in seen_ids:
                        seen_ids.add(ep['episode_id'])
                        unique_results.append(ep)
                
                return unique_results[:limit]
                
        except Exception as e:
            logger.error(f"Failed to query episode memory: {e}")
            return []
    
    def _vector_search_actor(
        self, 
        session, 
        namespace: str, 
        actor_name: str,
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = EntityEmbedder.encode(actor_name)
            result = session.run(
                """
                CALL db.index.vector.queryNodes('actor_embedding_index', $top_k, $embedding)
                YIELD node AS a, score
                WHERE a.namespace = $namespace AND score >= $threshold
                MATCH (e:Episode)-[:INVOLVES]->(a)
                WHERE e.namespace = $namespace
                RETURN DISTINCT e.episode_id AS episode_id,
                       e.summary AS summary,
                       e.content AS content,
                       e.timestamp AS timestamp,
                       e.event_type AS event_type,
                       a.name AS matched_actor,
                       score AS similarity
                ORDER BY score DESC
                LIMIT $limit
                """,
                embedding=query_embedding,
                namespace=namespace,
                top_k=limit * 2,
                threshold=threshold,
                limit=limit
            )
            
            episodes = []
            for record in result:
                episodes.append({
                    "episode_id": record["episode_id"],
                    "summary": record["summary"],
                    "content": record["content"],
                    "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                    "event_type": record["event_type"],
                    "matched_entity": record["matched_actor"],
                    "similarity": record["similarity"],
                    "match_type": "actor_vector"
                })
            return episodes
            
        except Exception:
            return []
    
    def _vector_search_location(
        self, 
        session, 
        namespace: str, 
        location_name: str,
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = EntityEmbedder.encode(location_name)
            result = session.run(
                """
                CALL db.index.vector.queryNodes('location_embedding_index', $top_k, $embedding)
                YIELD node AS l, score
                WHERE l.namespace = $namespace AND score >= $threshold
                MATCH (e:Episode)-[:OCCURS_AT]->(l)
                WHERE e.namespace = $namespace
                RETURN DISTINCT e.episode_id AS episode_id,
                       e.summary AS summary,
                       e.content AS content,
                       e.timestamp AS timestamp,
                       e.event_type AS event_type,
                       l.name AS matched_location,
                       score AS similarity
                ORDER BY score DESC
                LIMIT $limit
                """,
                embedding=query_embedding,
                namespace=namespace,
                top_k=limit * 2,
                threshold=threshold,
                limit=limit
            )
            
            episodes = []
            for record in result:
                episodes.append({
                    "episode_id": record["episode_id"],
                    "summary": record["summary"],
                    "content": record["content"],
                    "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                    "event_type": record["event_type"],
                    "matched_entity": record["matched_location"],
                    "similarity": record["similarity"],
                    "match_type": "location_vector"
                })
            return episodes
            
        except Exception:
            return []
    
    def _vector_search_person(
        self, 
        session, 
        namespace: str, 
        person_name: str, 
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = EntityEmbedder.encode(person_name)
            result = session.run(
                """
                CALL db.index.vector.queryNodes('person_embedding_index', $top_k, $embedding)
                YIELD node AS p, score
                WHERE p.namespace = $namespace AND score >= $threshold
                MATCH (e:Episode)-[:MENTIONS]->(p)
                WHERE e.namespace = $namespace
                RETURN DISTINCT e.episode_id AS episode_id,
                       e.summary AS summary,
                       e.content AS content,
                       e.timestamp AS timestamp,
                       e.event_type AS event_type,
                       p.name AS matched_person,
                       score AS similarity
                ORDER BY score DESC
                LIMIT $limit
                """,
                embedding=query_embedding,
                namespace=namespace,
                top_k=limit * 2, 
                threshold=threshold,
                limit=limit
            )
            
            episodes = []
            for record in result:
                episodes.append({
                    "episode_id": record["episode_id"],
                    "summary": record["summary"],
                    "content": record["content"],
                    "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                    "event_type": record["event_type"],
                    "matched_entity": record["matched_person"],
                    "similarity": record["similarity"],
                    "match_type": "person_vector"
                })
            return episodes
            
        except Exception:
            return []
    
    def _vector_search_topic(
        self, 
        session, 
        namespace: str, 
        topic_name: str, 
        limit: int,
        threshold: float
    ) -> List[Dict[str, Any]]:
        try:
            query_embedding = EntityEmbedder.encode(topic_name)
            result = session.run(
                """
                CALL db.index.vector.queryNodes('topic_embedding_index', $top_k, $embedding)
                YIELD node AS t, score
                WHERE t.namespace = $namespace AND score >= $threshold
                MATCH (e:Episode)-[:ABOUT]->(t)
                WHERE e.namespace = $namespace
                RETURN DISTINCT e.episode_id AS episode_id,
                       e.summary AS summary,
                       e.content AS content,
                       e.timestamp AS timestamp,
                       e.event_type AS event_type,
                       t.name AS matched_topic,
                       score AS similarity
                ORDER BY score DESC
                LIMIT $limit
                """,
                embedding=query_embedding,
                namespace=namespace,
                top_k=limit * 2,
                threshold=threshold,
                limit=limit
            )
            
            episodes = []
            for record in result:
                episodes.append({
                    "episode_id": record["episode_id"],
                    "summary": record["summary"],
                    "content": record["content"],
                    "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                    "event_type": record["event_type"],
                    "matched_entity": record["matched_topic"],
                    "similarity": record["similarity"],
                    "match_type": "topic_vector"
                })
            return episodes
            
        except Exception:
            return []
    
    def _time_range_query(
        self,
        session,
        namespace: str,
        start_time: Optional[str],
        end_time: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        query = "MATCH (e:Episode {namespace: $namespace})"
        params = {"namespace": namespace, "limit": limit}

        conditions = []
        if start_time:
            conditions.append("datetime(e.timestamp) >= datetime($start_time)")
            params["start_time"] = start_time
        if end_time:
            conditions.append("datetime(e.timestamp) <= datetime($end_time)")
            params["end_time"] = end_time
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += """
            RETURN e.episode_id AS episode_id,
                   e.summary AS summary,
                   e.content AS content,
                   e.timestamp AS timestamp,
                   e.event_type AS event_type
            ORDER BY e.timestamp DESC
            LIMIT $limit
        """
        
        result = session.run(query, **params)
        episodes = []
        for record in result:
            episodes.append({
                "episode_id": record["episode_id"],
                "summary": record["summary"],
                "content": record["content"],
                "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                "event_type": record["event_type"],
                "similarity": 0.5,  
                "match_type": "time_range"
            })
        return episodes
    
    def _structured_query(
        self,
        session,
        namespace: str,
        actor_name: Optional[str],
        location_name: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        results = []

        if actor_name:
            try:
                result = session.run(
                    """
                    MATCH (e:Episode {namespace: $namespace})
                    MATCH (e)-[:INVOLVES]->(a:Actor {namespace: $namespace})
                    WHERE TOLOWER(a.name) CONTAINS TOLOWER($actor_name)
                       OR TOLOWER($actor_name) CONTAINS TOLOWER(a.name)
                    RETURN DISTINCT e.episode_id AS episode_id,
                           e.summary AS summary,
                           e.content AS content,
                           e.timestamp AS timestamp,
                           e.event_type AS event_type,
                           a.name AS matched_actor,
                           0.7 AS similarity
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                    """,
                    namespace=namespace,
                    actor_name=actor_name,
                    limit=limit
                )
                
                for record in result:
                    results.append({
                        "episode_id": record["episode_id"],
                        "summary": record["summary"],
                        "content": record["content"],
                        "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                        "event_type": record["event_type"],
                        "matched_entity": record["matched_actor"],
                        "similarity": record["similarity"],
                        "match_type": "actor_fallback"
                    })
            except Exception:
                pass

        if location_name:
            try:
                result = session.run(
                    """
                    MATCH (e:Episode {namespace: $namespace})
                    MATCH (e)-[:OCCURS_AT]->(l:Location {namespace: $namespace})
                    WHERE TOLOWER(l.name) CONTAINS TOLOWER($location_name)
                       OR TOLOWER($location_name) CONTAINS TOLOWER(l.name)
                    RETURN DISTINCT e.episode_id AS episode_id,
                           e.summary AS summary,
                           e.content AS content,
                           e.timestamp AS timestamp,
                           e.event_type AS event_type,
                           l.name AS matched_location,
                           0.7 AS similarity
                    ORDER BY e.timestamp DESC
                    LIMIT $limit
                    """,
                    namespace=namespace,
                    location_name=location_name,
                    limit=limit
                )
                
                for record in result:
                    results.append({
                        "episode_id": record["episode_id"],
                        "summary": record["summary"],
                        "content": record["content"],
                        "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                        "event_type": record["event_type"],
                        "matched_entity": record["matched_location"],
                        "similarity": record["similarity"],
                        "match_type": "location_fallback"
                    })
            except Exception:
                pass
        
        return results
    
    def _fallback_string_match(
        self,
        session,
        namespace: str,
        person_name: Optional[str],
        topic_name: Optional[str],
        limit: int
    ) -> List[Dict[str, Any]]:
        query = "MATCH (e:Episode {namespace: $namespace})"
        params = {"namespace": namespace, "limit": limit}
        
        if person_name:
            query += """
                MATCH (e)-[:MENTIONS]->(p:Person {namespace: $namespace})
                WHERE TOLOWER(p.name) CONTAINS TOLOWER($person_name)
            """
            params["person_name"] = person_name
        
        if topic_name:
            query += """
                MATCH (e)-[:ABOUT]->(t:Topic {namespace: $namespace})
                WHERE TOLOWER(t.name) CONTAINS TOLOWER($topic_name)
                  OR TOLOWER($topic_name) CONTAINS TOLOWER(t.name)
            """
            params["topic_name"] = topic_name
        
        query += """
            RETURN e.episode_id AS episode_id,
                   e.summary AS summary,
                   e.content AS content,
                   e.timestamp AS timestamp,
                   e.event_type AS event_type
            ORDER BY e.timestamp DESC
            LIMIT $limit
        """
        
        result = session.run(query, **params)
        episodes = []
        for record in result:
            episodes.append({
                "episode_id": record["episode_id"],
                "summary": record["summary"],
                "content": record["content"],
                "timestamp": str(record["timestamp"]) if record["timestamp"] else None,
                "event_type": record["event_type"],
                "similarity": 0.7, 
                "match_type": "string_fallback"
            })
        return episodes
    
    def close(self):
        if self.driver:
            self.driver.close()
