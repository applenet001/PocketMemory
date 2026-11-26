from .base import MemBaseConfig


class HGPConfig(MemBaseConfig):
    checkpoint: str
    run_id: str
    env_file: str
    classifier_threshold: float = 0.9
    semantic_database: str = 'experiments-semantic-qwen'
    episodic_database: str = 'experiments-episode-qwen'

