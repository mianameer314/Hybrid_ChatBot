"""
RAG (Retrieval-Augmented Generation) API endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
import logging

from app.core.database import get_db
from app.services.rag_system import rag_system
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str
    k: Optional[int] = 5
    threshold: Optional[float] = 0.7

class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_documents: List[Dict[str, Any]]
    total_results: int
    filtered_results: int

@router.post("/query", response_model=QueryResponse)
async def query_rag_system(request: QueryRequest):
    """
    Query the RAG system for relevant information
    """
    try:
        result = await rag_system.query(
            query=request.query,
            k=request.k,
            threshold=request.threshold
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.get("/statistics")
async def get_rag_statistics():
    """
    Get RAG system statistics
    """
    try:
        stats = await rag_system.get_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get RAG statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@router.post("/initialize")
async def initialize_rag_system():
    """
    Initialize or reinitialize the RAG system
    """
    try:
        await rag_system.initialize()
        return {"status": "initialized", "message": "RAG system initialized successfully"}
        
    except Exception as e:
        logger.error(f"RAG initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@router.get("/health")
async def rag_health_check():
    """
    Check RAG system health
    """
    try:
        # Check if system is initialized
        if not rag_system.is_initialized:
            return {
                "status": "not_initialized",
                "message": "RAG system not initialized",
                "initialized": False
            }
        
        # Get basic statistics
        stats = await rag_system.get_statistics()
        
        return {
            "status": "healthy",
            "message": "RAG system is running",
            "initialized": True,
            "statistics": stats
        }
        
    except Exception as e:
        logger.error(f"RAG health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"RAG system error: {str(e)}",
            "initialized": rag_system.is_initialized
        }
