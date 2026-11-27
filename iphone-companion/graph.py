"""Experiment graph views and isolated writes to the usermemory database."""
from pathlib import Path
import os
from urllib.parse import urlsplit
from memory_draft import memory_evidence, memory_labels

ROOT = Path(__file__).resolve().parents[1]
DATABASES = ('semantic', 'episodic')
PERSONAL_DATABASE = 'usermemory'
PERSONAL_VIEWS = ('personal-semantic', 'personal-episodic')
FIELDS = ('name', 'title', 'content', 'text', 'timestamp', 'episode_id', 'user_id', 'namespace', 'relation', 'value', 'type')


class LocalGraph:
    def connection_settings(self):
        values = {}
        env_file = ROOT / 'config/local.env'
        if env_file.exists():
            for line in env_file.read_text(encoding='utf-8-sig').splitlines():
                key, sep, value = line.partition('=')
                if sep and not key.strip().startswith('#'):
                    values[key.strip()] = value.strip().strip('\"\'')
        uri = os.getenv('NEO4J_URI') or values.get('NEO4J_URI') or 'neo4j://127.0.0.1:7687'
        if urlsplit(uri).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('本地模式只允许连接这台电脑上的 Neo4j')
        username = os.getenv('NEO4J_USERNAME') or values.get('NEO4J_USERNAME')
        password = os.getenv('NEO4J_PASSWORD') or values.get('NEO4J_PASSWORD')
        if not username or not password:
            raise ValueError('请先配置 config/local.env 中的 Neo4j 凭据')
        return uri, username, password

    def driver(self):
        from neo4j import GraphDatabase

        uri, username, password = self.connection_settings()
        return GraphDatabase.driver(uri, auth=(username, password), connection_timeout=3,
                                   connection_acquisition_timeout=5, max_transaction_retry_time=0)

    def write_upload(self, namespace, item):
        from neo4j import Query

        result = item['classification']
        routes = [kind for kind in memory_labels(result) if kind in {'semantic', 'episodic'}]
        route = routes[0]
        label = {'semantic': 'SemanticCandidate', 'episodic': 'ImportedEpisode'}[route]
        identity = item['id']
        with self.driver() as driver, driver.session(database=PERSONAL_DATABASE) as session:
            session.run(Query(f'''
                OPTIONAL MATCH (original:PocketMemoryUpload {{namespace:$scope, source_id:$id}})
                WHERE original:SemanticCandidate OR original:ImportedEpisode
                WITH original ORDER BY original.route=$route DESC, elementId(original) LIMIT 1
                CALL (original) {{
                    WITH original WHERE original IS NOT NULL
                    RETURN original AS memory
                    UNION
                    WITH original WHERE original IS NULL
                    MERGE (memory:PocketMemoryUpload:{label} {{namespace:$scope, source_id:$id}})
                    RETURN memory
                }}
                REMOVE memory:SemanticCandidate:ImportedEpisode
                SET memory.route=$route, memory.routes=$routes,
                    memory.title=$title, memory.saved_at=$saved,
                    memory.content=$content, memory.confidence=$confidence, memory.source_app=$app
                FOREACH (_ IN CASE WHEN 'semantic' IN $routes THEN [1] ELSE [] END | SET memory:SemanticCandidate)
                FOREACH (_ IN CASE WHEN 'episodic' IN $routes THEN [1] ELSE [] END | SET memory:ImportedEpisode)
            ''', timeout=6), scope=namespace, id=identity, route=route, routes=routes, title=item['title'],
                        saved=item['created'], content=memory_evidence(item) or item['title'],
                        confidence=result.get('confidence'), app=item.get('source_app') or None).consume()

    def consolidate_upload_sources(self):
        """Remove legacy Companion source nodes whose data is already on their memory node."""
        from neo4j import Query

        with self.driver() as driver, driver.session(database=PERSONAL_DATABASE) as session:
            return session.run(Query('''
                MATCH (n:PocketMemoryUpload)-[link:HAS_SOURCE]->(s:PocketMemoryUpload:UploadSource)
                WHERE (n:SemanticCandidate OR n:ImportedEpisode)
                    AND n.namespace=s.namespace AND n.source_id=s.source_id
                    AND all(label IN labels(s) WHERE label IN ['UploadSource', 'PocketMemoryUpload'])
                    AND all(key IN keys(s) WHERE key IN ['namespace', 'source_id', 'title', 'route', 'saved_at', 'source_app']
                        AND s[key]=n[key])
                    AND properties(link) = {}
                    AND NOT EXISTS { MATCH (s)-[other]-() WHERE other <> link }
                DELETE link, s
            ''', timeout=6)).consume().counters.nodes_deleted

    def delete_upload(self, namespace, identity):
        from neo4j import Query

        with self.driver() as driver, driver.session(database=PERSONAL_DATABASE) as session:
            session.run(Query('MATCH (n:PocketMemoryUpload {namespace:$scope, source_id:$id}) DETACH DELETE n', timeout=6),
                        scope=namespace, id=identity).consume()

    def experiment_facts(self, user, namespace):
        from neo4j import Query, READ_ACCESS

        query = Query('''
            MATCH (:User {user_id:$user, namespace:$scope})-[r]->(e:Entity {namespace:$scope})
            RETURN type(r) AS relation, e.name AS entity,
                   toString(coalesce(r.last_seen_at, r.created_at)) AS date,
                   r.evidence_episode_ids AS evidence
            ORDER BY date DESC, relation, entity
        ''', timeout=5)
        with self.driver() as driver, driver.session(database='semantic', default_access_mode=READ_ACCESS) as session:
            return session.run(query, user=user, scope=namespace).data()

    def experiment_episode_scopes(self, episode_ids):
        from neo4j import Query, READ_ACCESS

        query = Query('''
            MATCH (e:Episode) WHERE e.episode_id IN $episode_ids AND e.namespace IS NOT NULL
            RETURN DISTINCT e.episode_id AS episode_id, e.namespace AS scope
        ''', timeout=5)
        with self.driver() as driver, driver.session(database='episodic', default_access_mode=READ_ACCESS) as session:
            return session.run(query, episode_ids=episode_ids).data()

    def view(self, database='episodic', namespace='', allowed_namespace=None):
        from neo4j import Query, READ_ACCESS
        if database not in (*DATABASES, *PERSONAL_VIEWS):
            raise ValueError('未知的图视图')
        route = database.removeprefix('personal-') if database in PERSONAL_VIEWS else None
        target = PERSONAL_DATABASE if route else database
        if allowed_namespace is not None and namespace and namespace != allowed_namespace:
            raise ValueError('不能访问其他用户的数据空间')
        route_filter = ' AND $route IN coalesce(n.routes, [n.route]) AND NOT n:UploadSource' if route else ''
        label = ':PocketMemoryUpload' if route else ''
        with self.driver() as driver, driver.session(database=target, default_access_mode=READ_ACCESS) as session:
            scopes = session.run(Query(
                f'MATCH (n{label}) WHERE n.namespace IS NOT NULL AND ($allowed IS NULL OR n.namespace=$allowed)'
                + route_filter + ' RETURN n.namespace AS name, count(*) AS count ORDER BY count DESC LIMIT 100',
                timeout=5), allowed=allowed_namespace, route=route).data()
            if not namespace:
                return {'database': database, 'scopes': scopes, 'nodes': [], 'edges': [], 'limit': 80}
            projection = ('n{.title,.content,.source_id,.source_app,.saved_at,.confidence,.route}' if route else
                          'n{.name,.title,.content,.text,.timestamp,.episode_id,.user_id,.namespace,.relation,.value,.type}')
            order = 'n.saved_at DESC, n.source_id, elementId(n)' if route else 'elementId(n)'
            edge_filter = (' AND $route IN coalesce(a.routes, [a.route]) AND $route IN coalesce(b.routes, [b.route])'
                           ' AND NOT a:UploadSource AND NOT b:UploadSource' if route else '')
            edge_order = 'a.saved_at DESC, elementId(r)' if route else 'elementId(r)'
            seed_edges = session.run(Query(
                f'MATCH (a{label})-[r]->(b{label}) WHERE a.namespace=$scope AND b.namespace=$scope'
                + edge_filter + f' RETURN elementId(a) AS source, elementId(b) AS target ORDER BY {edge_order} LIMIT 40',
                timeout=5), scope=namespace, route=route).data()
            # Forty edges require at most eighty endpoints, the node budget of a graph preview.
            endpoint_ids = list(dict.fromkeys(identity for edge in seed_edges
                                              for identity in (edge['source'], edge['target'])))
            nodes = session.run(Query(
                f'MATCH (n{label}) WHERE n.namespace=$scope' + route_filter
                + f' RETURN elementId(n) AS id, labels(n) AS labels, {projection} AS properties'
                + f' ORDER BY CASE WHEN elementId(n) IN $endpoints THEN 0 ELSE 1 END, {order} LIMIT 80',
                timeout=5), scope=namespace, route=route, endpoints=endpoint_ids).data()
            ids = [node['id'] for node in nodes]
            edges = session.run(Query('MATCH (a)-[r]->(b) WHERE elementId(a) IN $ids AND elementId(b) IN $ids AND a.namespace=$scope AND b.namespace=$scope RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS type LIMIT 160', timeout=5), ids=ids, scope=namespace).data()
            for node in nodes:
                node['properties'] = {key: str(value)[:2000] for key, value in node['properties'].items() if value is not None}
                props = node['properties']
                node['label'] = next((props[key] for key in ('name', 'title', 'content', 'text', 'episode_id', 'timestamp') if props.get(key)), '/'.join(node['labels']))[:50]
            return {'database': database, 'scopes': scopes, 'nodes': nodes, 'edges': edges, 'limit': 80}
