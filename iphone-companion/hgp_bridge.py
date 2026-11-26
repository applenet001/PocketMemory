"""Explicitly import a companion text export into the existing original HGP engine."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def records(export):
    if export.get('schema') != 'pocketmemory.share.v1':
        raise ValueError('Unsupported export schema')
    result = []
    for item in export['items']:
        content = '\n'.join(str(item.get(key, '')) for key in ('title', 'text', 'note', 'url')).strip()
        if content:
            result.append(dict(id='iphone-' + item['id'], name='iPhone user-shared content', role='user',
                               content=content, timestamp=item['created']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export', type=Path)
    parser.add_argument('--user-id', required=True)
    parser.add_argument('--apply', action='store_true', help='Write into HGP; otherwise validate only')
    parser.add_argument('--allow-cloud-extraction', action='store_true', help='Explicitly allow the existing HGP DeepSeek extractor to receive shared text')
    args = parser.parse_args()
    values = records(json.loads(args.export.read_text(encoding='utf-8-sig')))
    if not args.apply:
        print(json.dumps({'valid_records': len(values), 'mode': 'dry-run'}))
        return
    if not args.allow_cloud_extraction:
        parser.error('Original HGP semantic/procedural storage requires its cloud LLM extractor. Local-only mode blocks this import; use the local inbox until a local extraction adapter is configured.')
    sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'MemBase')]
    from hgp_original_layer import OriginalHGPConfig, OriginalHGPLayer
    from membase.model_types.dataset import Message
    import hashlib
    scope = hashlib.sha256(args.user_id.encode()).hexdigest()[:20]
    config = OriginalHGPConfig(user_id=args.user_id,
        save_dir=str(ROOT / 'outputs/iphone-hgp' / scope), graph_namespace='iphone-shares-v1',
        adaptive_updates=False, allow_llm_fallback=False, allow_llm_extraction=True)
    layer = OriginalHGPLayer(config)
    try:
        layer.add_messages([Message(**value) for value in values])
        layer.save_memory()
        print(json.dumps({'imported_records': len(values), 'namespace': 'iphone-shares-v1'}))
    finally:
        layer.close()


if __name__ == '__main__':
    main()
