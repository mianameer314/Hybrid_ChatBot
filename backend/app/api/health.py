"""
Health check and system status endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import psutil
import sys
import logging

from app.core.database import get_db, check_db_health
from app.core.cache import get_cache
from app.services.agent_system import agent_system

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "message": "Agentic Chatbot Backend is running"
    }

@router.get("/status")
async def system_status(db: Session = Depends(get_db)):
    """Comprehensive system status"""
    try:
        # Database health
        db_health = check_db_health()
        
        # Cache health
        cache = get_cache()
        cache_stats = cache.get_stats()
        
        # Agent system status
        agent_status = await agent_system.get_system_status()
        
        # System resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "database": db_health,
            "cache": cache_stats,
            "agent_system": agent_status,
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "used_percent": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "used_percent": round((disk.used / disk.total) * 100, 2)
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@router.get("/readiness")
async def readiness_check():
    """Kubernetes-style readiness probe"""
    try:
        # Check critical services
        db_health = check_db_health()
        if db_health["status"] != "healthy":
            raise HTTPException(status_code=503, detail="Database not ready")
        
        cache = get_cache()
        if not cache.is_available():
            raise HTTPException(status_code=503, detail="Cache not ready")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

@router.get("/liveness")
async def liveness_check():
    """Kubernetes-style liveness probe"""
    return {"status": "alive", "timestamp": datetime.now().isoformat()}
