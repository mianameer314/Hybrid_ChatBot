"""
FastAPI Backend for Agentic Chatbot
Features: RAG, Multi-LLM, Sentiment Analysis, Database Integration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging
from datetime import datetime

from app.core.config import settings
from app.core.database import engine, Base, run_migrations
from app.core.logging_config import setup_logging
from app.api import chat, documents, rag, health

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Agentic Chatbot Backend...")
    try:
        # Run database migrations
        run_migrations()
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agentic Chatbot Backend...")

# Create FastAPI app
app = FastAPI(
    title="Agentic Chatbot Backend",
    description="Advanced AI Chatbot with RAG, Multi-LLM support, and Sentiment Analysis",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Agentic Chatbot Backend API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Multi-LLM Support (OpenAI, Gemini, HuggingFace)",
            "RAG System (PDFs, Web, Database)",
            "Advanced Sentiment Analysis",
            "LangChain Agent Pipeline",
            "Conversation Memory",
            "Vector Embeddings"
        ]
    }

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(rag.router, prefix="/rag", tags=["RAG System"])

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
