from .base import MemBaseConfig
from pydantic import Field


class LongContextConfig(MemBaseConfig):
    """Configuration for the long-context based memory (online version).""" 

    message_separator: str = Field(
        default="\n",
        description="Separator used to join messages into a document."
    )
    context_window: int = Field(
        default=128_000,
        description=(
            "Long context window size. If the context window is exceeded, "
            "the earliest messages will be dropped."
        ),
        gt=0,
    )
    llm_model: str = Field(
        ...,
        min_length=1,
        description=(
            "Required tokenizer identifier for token counting only; not an inference model. "
            "An identifier from HuggingFace Hub or a local model path is also supported."
        ),
    )

