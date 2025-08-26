from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API keys
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # Database (PostgreSQL)
    DATABASE_URL: str = "postgresql://postgres:mypass@localhost:5432/chatbot_agent"

    # Redis (for caching / short-term memory)
    REDIS_URL: str = "redis://localhost:6379/0"

    API_URL: str | None = None

    # App metadata
    APP_TITLE: str = "AI Chatbot (Offline + Gemini + Memory)"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
