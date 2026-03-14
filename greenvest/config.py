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
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    POSTGRES_URL: str = field(default_factory=lambda: os.getenv("POSTGRES_URL", "postgresql://localhost:5432/greenvest"))
    WEAVIATE_URL: str = field(default_factory=lambda: os.getenv("WEAVIATE_URL", "http://localhost:8080"))


settings = Settings()
