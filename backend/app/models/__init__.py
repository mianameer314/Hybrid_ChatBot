"""
Database Models for Agentic Chatbot
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, JSON, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class ChatSession(Base):
    """Chat session model"""
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=True)  # Optional user identification
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

class ChatMessage(Base):
    """Chat message model with sentiment analysis"""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    
    # Sentiment Analysis Results
    sentiment_label = Column(String(20), nullable=True)  # positive, negative, neutral
    sentiment_score = Column(Float, nullable=True)  # confidence score
    emotion = Column(String(30), nullable=True)  # joy, anger, fear, etc.
    
    # Metadata
    token_count = Column(Integer, nullable=True)
    model_used = Column(String(50), nullable=True)  # which LLM was used
    processing_time = Column(Float, nullable=True)  # response time in seconds
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

class Document(Base):
    """Document storage for RAG system"""
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    original_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)  # pdf, txt, docx, etc.
    file_size = Column(Integer, nullable=False)  # in bytes
    file_path = Column(String(500), nullable=False)
    
    # Content and processing
    content = Column(Text, nullable=True)  # extracted text content
    summary = Column(Text, nullable=True)  # AI-generated summary
    
    # Metadata
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    processing_status = Column(String(20), default="pending")  # pending, processing, completed, failed
    
    # Document metadata
    title = Column(String(500), nullable=True)
    author = Column(String(255), nullable=True)
    creation_date = Column(DateTime, nullable=True)
    language = Column(String(10), nullable=True)
    
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    """Document chunks for vector embeddings"""
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, ForeignKey("documents.id"), nullable=False)
    
    # Chunk content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # order in document
    start_char = Column(Integer, nullable=True)  # start position in original document
    end_char = Column(Integer, nullable=True)  # end position in original document
    
    # Vector embedding (stored as JSON array)
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

class WebSource(Base):
    """Web sources for RAG system"""
    __tablename__ = "web_sources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    url = Column(String(1000), nullable=False, unique=True)
    domain = Column(String(255), nullable=False)
    title = Column(String(500), nullable=True)
    
    # Content
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    
    # Scraping metadata
    scraped_at = Column(DateTime, nullable=True)
    last_updated = Column(DateTime, nullable=True)
    status_code = Column(Integer, nullable=True)
    content_type = Column(String(100), nullable=True)
    
    # SEO/Content metadata
    meta_description = Column(Text, nullable=True)
    keywords = Column(JSON, nullable=True)
    language = Column(String(10), nullable=True)
    
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("WebChunk", back_populates="source", cascade="all, delete-orphan")

class WebChunk(Base):
    """Web content chunks for vector embeddings"""
    __tablename__ = "web_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(String, ForeignKey("web_sources.id"), nullable=False)
    
    # Chunk content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    selector = Column(String(200), nullable=True)  # CSS selector if available
    
    # Vector embedding
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    
    # Relationships
    source = relationship("WebSource", back_populates="chunks")

class KnowledgeBase(Base):
    """Structured knowledge base entries"""
    __tablename__ = "knowledge_base"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    tags = Column(JSON, nullable=True)  # array of strings
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    priority = Column(Integer, default=0)  # higher = more important
    
    # Vector embedding for semantic search
    embedding = Column(JSON, nullable=True)
    embedding_model = Column(String(100), nullable=True)
    
    # Usage statistics
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, nullable=True)
    
    metadata = Column(JSON, nullable=True)
