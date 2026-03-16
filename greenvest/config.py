import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    USE_MOCK_LLM: bool = field(
        default_factory=lambda: os.getenv("USE_MOCK_LLM", "true").lower() != "false"
    )
    ANTHROPIC_API_KEY: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    GOOGLE_API_KEY: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    QDRANT_URL: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    POSTGRES_URL: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", "postgresql://localhost:5432/greenvest"))
    OLLAMA_BASE_URL: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    OLLAMA_ROUTER_MODEL: str = field(default_factory=lambda: os.getenv("OLLAMA_ROUTER_MODEL", "llama3.2:latest"))
    OLLAMA_SYNTHESIZER_MODEL: str = field(default_factory=lambda: os.getenv("OLLAMA_SYNTHESIZER_MODEL", "llama3:latest"))
    OLLAMA_JUDGE_MODEL: str = field(default_factory=lambda: os.getenv("OLLAMA_JUDGE_MODEL", "gemma2:9b"))
    LANGFUSE_PUBLIC_KEY: str = field(default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY", ""))
    LANGFUSE_SECRET_KEY: str = field(default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY", ""))
    LANGFUSE_HOST: str = field(default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"))


settings = Settings()
