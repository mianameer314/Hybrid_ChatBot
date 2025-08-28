"""
Document and RAG management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import uuid
import logging
from datetime import datetime

from app.core.database import get_db
from app.services.rag_system import rag_system
from app.models import Document as DBDocument
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    processing_time: float

class WebScrapeRequest(BaseModel):
    urls: List[str]
    max_pages: Optional[int] = 10

class KnowledgeEntryRequest(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None
    tags: Optional[List[str]] = None

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    filename: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document for RAG
    """
    start_time = datetime.now()
    
    try:
        # Validate file type
        allowed_extensions = ['.pdf', '.txt', '.docx', '.md']
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not supported. Allowed: {allowed_extensions}"
            )
        
        # Generate unique filename
        upload_filename = filename or file.filename
        unique_filename = f"{uuid.uuid4()}_{upload_filename}"
        file_path = os.path.join("uploads", unique_filename)
        
        # Ensure upload directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        file_size = len(content)
        logger.info(f"File uploaded: {upload_filename} ({file_size} bytes)")
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_path=file_path,
            filename=upload_filename,
            file_type=file_ext[1:]  # Remove the dot
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DocumentUploadResponse(
            document_id=unique_filename,
            filename=upload_filename,
            file_type=file_ext[1:],
            file_size=file_size,
            status="processing",
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def process_document_background(file_path: str, filename: str, file_type: str):
    """Background task to process uploaded document"""
    try:
        document_id = await rag_system.add_document(
            file_path=file_path,
            filename=filename,
            file_type=file_type
        )
        logger.info(f"Document processed successfully: {filename} -> {document_id}")
    except Exception as e:
        logger.error(f"Document processing failed for {filename}: {e}")

@router.post("/web-scrape")
async def scrape_web_content(
    request: WebScrapeRequest,
    background_tasks: BackgroundTasks
):
    """
    Scrape web content and add to RAG system
    """
    try:
        if len(request.urls) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 URLs allowed per request")
        
        # Process web scraping in background
        background_tasks.add_task(
            scrape_urls_background,
            urls=request.urls
        )
        
        return {
            "status": "processing",
            "urls_count": len(request.urls),
            "message": "Web scraping started in background"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Web scraping failed: {e}")
        raise HTTPException(status_code=500, detail=f"Web scraping failed: {str(e)}")

async def scrape_urls_background(urls: List[str]):
    """Background task to scrape URLs"""
    try:
        result_ids = await rag_system.add_web_content(urls)
        logger.info(f"Web scraping completed: {len(result_ids)} sources added")
    except Exception as e:
        logger.error(f"Web scraping background task failed: {e}")

@router.post("/knowledge-base")
async def add_knowledge_entry(request: KnowledgeEntryRequest):
    """
    Add an entry to the structured knowledge base
    """
    try:
        entry_id = await rag_system.add_knowledge_entry(
            question=request.question,
            answer=request.answer,
            category=request.category,
            tags=request.tags
        )
        
        return {
            "entry_id": entry_id,
            "status": "added",
            "question": request.question[:100] + "..." if len(request.question) > 100 else request.question
        }
        
    except Exception as e:
        logger.error(f"Knowledge base entry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add entry: {str(e)}")

@router.get("/list")
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """
    List uploaded documents
    """
    try:
        documents = db.query(DBDocument)\
                     .offset(offset)\
                     .limit(limit)\
                     .all()
        
        total_count = db.query(DBDocument).count()
        
        return {
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.original_name,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "upload_date": doc.upload_date.isoformat(),
                    "processing_status": doc.processing_status,
                    "title": doc.title,
                    "author": doc.author
                }
                for doc in documents
            ],
            "total_count": total_count,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@router.get("/{document_id}")
async def get_document(document_id: str, db: Session = Depends(get_db)):
    """
    Get document details
    """
    try:
        document = db.query(DBDocument).filter(DBDocument.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "filename": document.original_name,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "upload_date": document.upload_date.isoformat(),
            "processing_status": document.processing_status,
            "processed_at": document.processed_at.isoformat() if document.processed_at else None,
            "title": document.title,
            "author": document.author,
            "summary": document.summary,
            "content_preview": document.content[:500] + "..." if document.content and len(document.content) > 500 else document.content
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(document_id: str, db: Session = Depends(get_db)):
    """
    Delete a document and its chunks
    """
    try:
        document = db.query(DBDocument).filter(DBDocument.id == document_id).first()
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete file if exists
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete from database (cascades to chunks)
        db.delete(document)
        db.commit()
        
        logger.info(f"Document deleted: {document_id}")
        
        return {
            "status": "deleted",
            "document_id": document_id,
            "filename": document.original_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
