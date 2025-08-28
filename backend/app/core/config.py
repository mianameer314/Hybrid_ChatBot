"""
Configuration Management for Agentic Chatbot
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # App settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Security
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"]
    SECRET_KEY: str = "your-secret-key-change-in-production"
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/chatbot_db"
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 300  # 5 minutes
    
    # LLM API Keys
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    
    # Model Settings
    DEFAULT_LLM: str = "gemini"  # openai, gemini, huggingface (changed to gemini due to better free tier)
    OPENAI_MODEL: str = "gpt-4o-mini"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-medium"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_DIMENSION: int = 384
    
    # RAG Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_RETRIEVAL_DOCS: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # File Upload Settings
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".md"]
    
    # Web Scraping Settings
    MAX_PAGES_PER_DOMAIN: int = 10
    SCRAPING_DELAY: float = 1.0
    
    # Sentiment Analysis
    SENTIMENT_MODEL: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings
settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
