"""Experiment graph views and isolated writes to the usermemory database."""
from pathlib import Path
import os
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATABASES = ('semantic', 'episodic')
PERSONAL_DATABASE = 'usermemory'
PERSONAL_VIEWS = ('personal-semantic', 'personal-episodic')
FIELDS = ('name', 'title', 'content', 'text', 'timestamp', 'episode_id', 'user_id', 'namespace', 'relation', 'value', 'type')


class LocalGraph:
    def driver(self):
        from neo4j import GraphDatabase
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
        return GraphDatabase.driver(uri, auth=(username, password), connection_timeout=3,
                                   connection_acquisition_timeout=5, max_transaction_retry_time=0)

    def write_upload(self, namespace, item):
        from neo4j import Query

        result = item['classification']
        route = result['label']
        label = {'semantic': 'SemanticCandidate', 'episodic': 'ImportedEpisode'}[route]
        identity = item['id']
        with self.driver() as driver, driver.session(database=PERSONAL_DATABASE) as session:
            session.run(Query(f'''
                MERGE (n:{label} {{namespace:$scope, source_id:$id}})
                SET n:PocketMemoryUpload, n.route=$route, n.title=$title, n.saved_at=$saved,
                    n.content=$content, n.confidence=$confidence
                MERGE (s:UploadSource {{namespace:$scope, source_id:$id}})
                SET s:PocketMemoryUpload, s.route=$route, s.title=$title, s.saved_at=$saved
                MERGE (n)-[:HAS_SOURCE]->(s)
            ''', timeout=6), scope=namespace, id=identity, route=route, title=item['title'],
                        saved=item['created'], content='\n'.join(filter(None, (item['text'], item['note']))),
                        confidence=result['confidence']).consume()

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

    def view(self, database='episodic', namespace='', allowed_namespace=None):
        from neo4j import Query, READ_ACCESS
        if database not in (*DATABASES, *PERSONAL_VIEWS):
            raise ValueError('未知的图视图')
        route = database.removeprefix('personal-') if database in PERSONAL_VIEWS else None
        target = PERSONAL_DATABASE if route else database
        if allowed_namespace is not None and namespace and namespace != allowed_namespace:
            raise ValueError('不能访问其他用户的数据空间')
        route_filter = ' AND n.route=$route' if route else ''
        with self.driver() as driver, driver.session(database=target, default_access_mode=READ_ACCESS) as session:
            scopes = session.run(Query(
                'MATCH (n) WHERE n.namespace IS NOT NULL AND ($allowed IS NULL OR n.namespace=$allowed)'
                + route_filter + ' RETURN n.namespace AS name, count(*) AS count ORDER BY count DESC LIMIT 100',
                timeout=5), allowed=allowed_namespace, route=route).data()
            if not namespace:
                return {'database': database, 'scopes': scopes, 'nodes': [], 'edges': [], 'limit': 80}
            projection = ('n{.title,.content,.source_id,.saved_at,.confidence,.route}' if route else
                          'n{.name,.title,.content,.text,.timestamp,.episode_id,.user_id,.namespace,.relation,.value,.type}')
            label = ':PocketMemoryUpload' if route else ''
            order = 'n.saved_at DESC, n.source_id, elementId(n)' if route else 'elementId(n)'
            edge_filter = ' AND a.route=$route AND b.route=$route' if route else ''
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
