"""Invoke the HGP evaluator's retrieval for one existing experiment user."""
from pathlib import Path
import sys
from threading import Lock

ROOT = Path(__file__).resolve().parents[1]
TOP_K = 5
_lock = Lock()


def search_run(config, folder, user, question, *, scope=None):
    with _lock:
        for module_path in (ROOT / 'scripts', ROOT / 'MemBase'):
            if str(module_path) not in sys.path:
                sys.path.append(str(module_path))
        from hgp_original_layer import OriginalHGPConfig, OriginalHGPLayer, build_hgp_qa_context

        settings = OriginalHGPConfig(**{**config, 'user_id': user, 'save_dir': str(folder)})
        layer = OriginalHGPLayer.open_for_search(settings, scope=scope)
        try:
            result, working = layer.search_matches(question, TOP_K)
            memories = layer.entries_for_matches(result, working)
            context = build_hgp_qa_context(memories) if memories else None
            return result, working, layer.query_extraction, context
        finally:
            layer.close()
