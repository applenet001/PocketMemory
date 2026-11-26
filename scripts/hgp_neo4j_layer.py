"""MemBase HGP layer backed by a user-scoped Neo4j memory graph."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from membase.configs.base import MemBaseConfig
from membase.layers.base import MemBaseLayer
from membase.model_types.memory import MemoryEntry
from pydantic import Field

from hgp3_router import AdaptiveRouter, deepseek_route
from train_hgp3_2200 import DATA, ROOT


METHOD = 'HGP'
CLASSES = {
    'semantic': 'HGPSemantic', 'episodic': 'HGPEpisodic',
    'procedural': 'HGPProcedural', 'working': 'HGPWorking',
}
SCHEMA = tuple(
    f'CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE'
    for name, label in (
        ('hgp_graph_key', 'HGPGraph'), ('hgp_source_key', 'HGPSource'),
        ('hgp_memory_key', 'HGPMemory'), ('hgp_step_key', 'HGPStep'),
    )
)


def settings(path):
    values = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        key, sep, value = line.partition('=')
        if sep and key.strip() and not key.lstrip().startswith('#'):
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def graph_driver(env_file):
    from neo4j import GraphDatabase

    values = settings(env_file)
    keys = ('NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD', 'NEO4J_DATABASE')
    params = [os.environ.get(key) or values.get(key) for key in keys]
    if not all(params):
        raise ValueError('Missing Neo4j settings: ' + ', '.join(key for key, value in zip(keys, params) if not value))
    uri, username, password, db = params
    return GraphDatabase.driver(uri, auth=(username, password), connection_timeout=5), db


def initialize_graph(env_file):
    driver, database = graph_driver(env_file)
    with driver, driver.session(database=database) as session:
        session.run('RETURN 1').consume()
        for query in SCHEMA:
            session.run(query).consume()


def id_for(*parts):
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


class HGPConfig(MemBaseConfig):
    user_id: str = 'guest'
    env_file: Path = ROOT / 'config/local.env'
    classifier_dir: Path = ROOT / 'outputs/hgp3-train-2200/classifier'
    training_data: Path = DATA
    graph_namespace: str = Field(default='mobilemem-hgp3', min_length=1)
    tau0: float = 0.9
    tau_min: float = 0.5
    beta: float = 0.05
    update_batch_size: int = 100
    adaptive_updates: bool = True
    allow_llm_fallback: bool = False

    def get_llm_models(self):
        return ['deepseek-flash'] if self.allow_llm_fallback else []


class HGPLayer(MemBaseLayer):
    def __init__(self, config: HGPConfig, router=None, driver=None, database=None):
        self.config = config
        self.scope = id_for(config.graph_namespace, config.user_id)
        self.router = router or AdaptiveRouter(
            config.classifier_dir, config.training_data, Path(config.save_dir) / 'adaptive-router.pt',
            fallback=deepseek_route if config.allow_llm_fallback else None,
            tau0=config.tau0, tau_min=config.tau_min, beta=config.beta,
            batch_size=config.update_batch_size, adaptive=config.adaptive_updates,
        )
        self.identity = id_for(self.router.identity, config.tau0, config.tau_min, config.beta,
                               config.update_batch_size, config.adaptive_updates, config.graph_namespace)
        self.driver, self.database = (driver, database) if driver else graph_driver(config.env_file)

    def _scope(self, tx):
        row = tx.run('''
            MERGE (g:HGPGraph {key: $scope})
            ON CREATE SET g.user_id = $user, g.namespace = $namespace, g.identity = $identity, g.ready = false
            RETURN g.identity AS identity
        ''', scope=self.scope, user=self.config.user_id, namespace=self.config.graph_namespace,
            identity=self.identity).single(strict=True)
        if row['identity'] != self.identity:
            raise ValueError('Graph namespace belongs to another HGP model or configuration')

    def _existing(self, session, source_key, evidence):
        record = session.run(
            'MATCH (s:HGPSource {key: $key}) RETURN s.evidence AS evidence', key=source_key,
        ).single()
        if record and record['evidence'] != evidence:
            raise ValueError('Source ID reused with different evidence')
        return record is not None

    def add_message(self, message, **kwargs):
        body = message.content
        if not isinstance(body, str):
            raise ValueError('HGP text memory requires string message content')
        if not body.strip():
            return
        source_id = str(message.id)
        source_key = id_for(self.scope, source_id)
        route_input = dict(id=source_id, text=body.strip(), role=message.role)
        time = datetime.fromisoformat(message.timestamp)
        timestamp = (time.astimezone(timezone.utc).replace(tzinfo=None) if time.tzinfo else time).isoformat()
        evidence = id_for(source_id, body, message.role, message.name, timestamp)
        with self.driver.session(database=self.database) as session:
            if self._existing(session, source_key, evidence):
                return
            result, vector = self.router.route(route_input)
            kinds = [kind for kind in (result['label'], 'working' if result['working_state'] else None) if kind]
            inserted = session.execute_write(
                self._write, source_key, source_id, evidence, timestamp, message.name, route_input,
                result, vector.tolist(), kinds,
            )
        if inserted:
            self.router.observe(route_input, result, vector)

    def _write(self, tx, source_key, source_id, evidence, timestamp, name, payload, result, vector, kinds):
        self._scope(tx)
        if self._existing(tx, source_key, evidence):
            return False
        tx.run('''
            MATCH (g:HGPGraph {key: $scope})
            CREATE (s:HGPSource {key: $key, id: $id, scope: $scope, evidence: $evidence, text: $text,
                timestamp: $timestamp, role: $role, name: $name, route: $route,
                confidence: $confidence, threshold: $threshold, uncertain: $uncertain,
                prompt_tokens: $prompt_tokens, completion_tokens: $completion_tokens})
            CREATE (g)-[:HGP_HAS_SOURCE]->(s)
        ''', scope=self.scope, key=source_key, id=source_id, evidence=evidence, text=payload['text'],
            timestamp=timestamp, role=payload['role'], name=name, route=result['source'],
            confidence=result['confidence'], threshold=result['threshold'], uncertain=result['uncertain'],
            prompt_tokens=result.get('usage', {}).get('prompt_tokens'),
            completion_tokens=result.get('usage', {}).get('completion_tokens')).consume()
        for kind in kinds:
            key = id_for(source_key, kind)
            tx.run(f'''
                MATCH (g:HGPGraph {{key: $scope}}), (s:HGPSource {{key: $source}})
                CREATE (m:HGPMemory:{CLASSES[kind]} {{key: $key, scope: $scope, kind: $kind,
                    text: $text, timestamp: $timestamp, vector: $vector, active: true}})
                CREATE (g)-[:HGP_HAS_MEMORY]->(m)
                CREATE (m)-[:HGP_EVIDENCE]->(s)
            ''', source=source_key, key=key, scope=self.scope, kind=kind, text=payload['text'],
                timestamp=timestamp, vector=vector).consume()
            if kind in ('episodic', 'working'):
                self._link_timeline(tx, key, kind)
            if kind == 'procedural':
                steps = re.findall(r'(?m)^\s*(?:\d{1,2}[.、)）]|[一二三四五六七八九十]+、)\s*([^\n]+)', payload['text'])
                self._link_steps(tx, key, steps)
        return True

    def _link_timeline(self, tx, key, kind):
        relation = 'HGP_NEXT_EVENT' if kind == 'episodic' else 'HGP_NEXT_STATE'
        tx.run(f'''
            MATCH (g:HGPGraph {{key: $scope}})-[:HGP_HAS_MEMORY]->(m:HGPMemory {{key: $key}})
            CALL (g,m) {{
                OPTIONAL MATCH (g)-[:HGP_HAS_MEMORY]->(p:HGPMemory {{kind: $kind}})
                WHERE p.timestamp < m.timestamp OR (p.timestamp = m.timestamp AND p.key < m.key)
                RETURN p ORDER BY p.timestamp DESC, p.key DESC LIMIT 1
            }}
            CALL (g,m) {{
                OPTIONAL MATCH (g)-[:HGP_HAS_MEMORY]->(n:HGPMemory {{kind: $kind}})
                WHERE n.timestamp > m.timestamp OR (n.timestamp = m.timestamp AND n.key > m.key)
                RETURN n ORDER BY n.timestamp, n.key LIMIT 1
            }}
            OPTIONAL MATCH (p)-[old:{relation}]->(n)
            FOREACH (_ IN CASE WHEN old IS NULL THEN [] ELSE [1] END | DELETE old)
            FOREACH (_ IN CASE WHEN p IS NULL THEN [] ELSE [1] END | MERGE (p)-[:{relation}]->(m))
            FOREACH (_ IN CASE WHEN n IS NULL THEN [] ELSE [1] END | MERGE (m)-[:{relation}]->(n))
        ''', scope=self.scope, key=key, kind=kind).consume()

    def _link_steps(self, tx, memory, steps):
        rows = [dict(key=id_for(memory, i), position=i, text=text.strip()) for i, text in enumerate(steps, 1)]
        if rows:
            tx.run('''
                MATCH (m:HGPMemory {key: $memory})
                UNWIND $steps AS row
                CREATE (step:HGPStep {key: row.key, position: row.position, text: row.text})
                CREATE (m)-[:HGP_HAS_STEP]->(step)
            ''', memory=memory, steps=rows).consume()

    def add_messages(self, messages, **kwargs):
        for message in messages:
            self.add_message(message, **kwargs)

    def retrieve(self, query, k=10, **kwargs):
        if k < 1:
            raise ValueError('k must be positive')
        vector = self.router.vector(query).tolist()
        with self.driver.session(database=self.database) as session:
            records = list(session.run('''
                MATCH (:HGPGraph {key: $scope})-[:HGP_HAS_MEMORY]->(m:HGPMemory {active: true})
                MATCH (m)-[:HGP_EVIDENCE]->(s:HGPSource)
                WITH m, s, vector.similarity.cosine(m.vector, $query_vector) AS score
                ORDER BY score DESC, m.key LIMIT $limit
                CALL (m) {
                    OPTIONAL MATCH (m)-[:HGP_HAS_STEP]->(step:HGPStep)
                    WITH step ORDER BY step.position
                    RETURN collect(step.text) AS steps
                }
                CALL (m) {
                    OPTIONAL MATCH (m)-[:HGP_NEXT_EVENT|HGP_NEXT_STATE]-(neighbor:HGPMemory {active: true})
                    WHERE neighbor.scope = $scope
                    WITH neighbor ORDER BY neighbor.timestamp, neighbor.key
                    RETURN collect(neighbor.timestamp + ' ' + neighbor.text)[..2] AS adjacent
                }
                RETURN m.key AS memory_id, m.kind AS kind, m.text AS text, s.id AS source_id,
                       m.timestamp AS timestamp, s.name AS name, score, steps, adjacent
            ''', scope=self.scope, query_vector=vector, limit=k))
        result = []
        for row in records:
            context = f'[{row["kind"]}] [{row["timestamp"]}] [{row["name"]}] {row["text"]}'
            if row['steps']:
                context += '\n' + '\n'.join(f'{i}. {text}' for i, text in enumerate(row['steps'], 1))
            if row['adjacent']:
                context += '\nAdjacent context:\n' + '\n'.join(row['adjacent'])
            metadata = {key: row[key] for key in ('memory_id', 'kind', 'source_id', 'timestamp', 'score')}
            result.append(MemoryEntry(content=row['text'], formatted_content=context, metadata=metadata))
        return result

    def delete(self, memory_id):
        with self.driver.session(database=self.database) as session:
            record = session.run('''
                MATCH (m:HGPMemory {key: $key, scope: $scope, active: true})
                SET m.active = false RETURN count(m) AS changed
            ''', key=memory_id, scope=self.scope).single(strict=True)
            return bool(record['changed'])

    def update(self, memory_id, **kwargs):
        text = kwargs.get('content')
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Supply nonempty content for the replacement memory')
        vector = self.router.vector(text.strip()).tolist()
        with self.driver.session(database=self.database) as session:
            record = session.run('''
                MATCH (m:HGPMemory {key: $key, scope: $scope, active: true})
                OPTIONAL MATCH (m)-[:HGP_HAS_STEP]->(step:HGPStep)
                WITH m, collect(step) AS obsolete
                FOREACH (step IN obsolete | DETACH DELETE step)
                SET m.text = $text, m.vector = $vector RETURN count(m) AS changed
            ''', key=memory_id, scope=self.scope, text=text.strip(), vector=vector).single(strict=True)
            return bool(record['changed'])

    def save_memory(self):
        with self.driver.session(database=self.database) as session:
            def commit(tx):
                self._scope(tx)
                tx.run('MATCH (g:HGPGraph {key: $scope}) SET g.ready = true', scope=self.scope).consume()
            session.execute_write(commit)

    def load_memory(self, user_id=None):
        if user_id is not None and user_id != self.config.user_id:
            return False
        with self.driver.session(database=self.database) as session:
            row = session.run('''
                MATCH (g:HGPGraph {key: $scope, ready: true}) RETURN g.identity AS identity
            ''', scope=self.scope).single()
            if row and row['identity'] != self.identity:
                raise ValueError('Graph namespace belongs to another HGP model or configuration')
            return row is not None

    def close(self):
        self.driver.close()


def register():
    from membase.configs import CONFIG_MAPPING
    from membase.layers import MEMORY_LAYERS_MAPPING

    CONFIG_MAPPING.register(METHOD, HGPConfig)
    MEMORY_LAYERS_MAPPING.register(METHOD, HGPLayer)
