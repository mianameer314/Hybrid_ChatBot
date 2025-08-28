"""
Chat API endpoints for the Agentic Chatbot
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
import uuid
import json
import asyncio
from datetime import datetime
from io import BytesIO

from app.core.database import get_db
from app.core.cache import get_cache
from app.core.config import settings
from app.services.agent_system import agent_system
from app.services.sentiment_analysis import analyze_sentiment
from app.services.llm_providers import llm_manager
from app.models import ChatSession, ChatMessage
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    llm_provider: Optional[str] = None
    use_agent: bool = True
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model_used: str
    processing_time: float
    sentiment_analysis: Optional[Dict[str, Any]] = None
    intermediate_steps: Optional[List[Dict[str, Any]]] = None
    
class SessionCreateRequest(BaseModel):
    title: Optional[str] = None
    user_id: Optional[str] = None

class SessionResponse(BaseModel):
    id: str
    title: Optional[str]
    created_at: str
    is_active: bool
    message_count: int

class MessageResponse(BaseModel):
    role: str
    content: str
    sentiment_label: Optional[str]
    emotion: Optional[str]
    created_at: str
    model_used: Optional[str]

@router.post("/message", response_model=ChatResponse)
async def send_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Send a message to the chatbot
    """
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Ensure session exists in database
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message
            )
            db.add(session)
            db.commit()
        
        # Process message through agent or direct LLM
        if request.use_agent:
            result = await agent_system.process_message(
                session_id=session_id,
                message=request.message,
                llm_provider=request.llm_provider,
                context=request.context
            )
        else:
            # Direct LLM call without agent
            start_time = datetime.now()
            
            # Get conversation history
            messages = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id
            ).order_by(ChatMessage.created_at.desc()).limit(10).all()
            
            # Format messages for LLM
            formatted_messages = []
            for msg in reversed(messages):
                formatted_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current message
            formatted_messages.append({
                "role": "user",
                "content": request.message
            })
            
            # Generate response
            response = await llm_manager.generate_response(
                messages=formatted_messages,
                provider_name=request.llm_provider
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'response': response,
                'processing_time': processing_time,
                'session_id': session_id,
                'model_used': request.llm_provider or settings.DEFAULT_LLM
            }
        
        # Analyze sentiment in background
        sentiment_task = background_tasks.add_task(
            analyze_user_message_sentiment,
            request.message,
            session_id
        )
        
        return ChatResponse(
            response=result['response'],
            session_id=result['session_id'],
            model_used=result['model_used'],
            processing_time=result['processing_time'],
            intermediate_steps=result.get('intermediate_steps', [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/stream")
async def stream_message(request: ChatRequest):
    """
    Stream a chat response
    """
    try:
        session_id = request.session_id or str(uuid.uuid4())
        
        async def generate():
            try:
                # For streaming, use direct LLM call
                provider = await llm_manager.get_provider(request.llm_provider)
                
                # Get recent messages
                with get_db() as db:
                    messages = db.query(ChatMessage).filter(
                        ChatMessage.session_id == session_id
                    ).order_by(ChatMessage.created_at.desc()).limit(10).all()
                    
                    formatted_messages = []
                    for msg in reversed(messages):
                        formatted_messages.append({
                            "role": msg.role,
                            "content": msg.content
                        })
                    
                    formatted_messages.append({
                        "role": "user",
                        "content": request.message
                    })
                
                # Stream response
                async for chunk in provider.stream_chat_completion(formatted_messages):
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming message: {str(e)}")

@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    db: Session = Depends(get_db)
):
    """
    Create a new chat session
    """
    try:
        session = ChatSession(
            title=request.title or "New Chat",
            user_id=request.user_id
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        return SessionResponse(
            id=session.id,
            title=session.title,
            created_at=session.created_at.isoformat(),
            is_active=session.is_active,
            message_count=0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating session: {str(e)}")

@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    user_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get list of chat sessions
    """
    try:
        query = db.query(ChatSession)
        
        if user_id:
            query = query.filter(ChatSession.user_id == user_id)
        
        sessions = query.filter(ChatSession.is_active == True)\
                        .order_by(ChatSession.updated_at.desc())\
                        .limit(limit).all()
        
        result = []
        for session in sessions:
            message_count = db.query(ChatMessage).filter(
                ChatMessage.session_id == session.id
            ).count()
            
            result.append(SessionResponse(
                id=session.id,
                title=session.title,
                created_at=session.created_at.isoformat(),
                is_active=session.is_active,
                message_count=message_count
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sessions: {str(e)}")

@router.get("/sessions/{session_id}/messages", response_model=List[MessageResponse])
async def get_session_messages(
    session_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """
    Get messages for a specific session
    """
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id
        ).order_by(ChatMessage.created_at.asc()).limit(limit).all()
        
        return [
            MessageResponse(
                role=msg.role,
                content=msg.content,
                sentiment_label=msg.sentiment_label,
                emotion=msg.emotion,
                created_at=msg.created_at.isoformat(),
                model_used=msg.model_used
            )
            for msg in messages
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting messages: {str(e)}")

@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Clear a chat session
    """
    try:
        success = await agent_system.clear_session(session_id)
        
        if success:
            return {"message": "Session cleared successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear session")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@router.get("/sessions/{session_id}/sentiment-summary")
async def get_session_sentiment_summary(
    session_id: str,
    db: Session = Depends(get_db)
):
    """
    Get sentiment analysis summary for a session
    """
    try:
        messages = db.query(ChatMessage).filter(
            ChatMessage.session_id == session_id,
            ChatMessage.role == "user",
            ChatMessage.sentiment_label.isnot(None)
        ).all()
        
        if not messages:
            return {"message": "No sentiment data available for this session"}
        
        # Calculate sentiment statistics
        sentiments = [msg.sentiment_label for msg in messages]
        scores = [msg.sentiment_score for msg in messages if msg.sentiment_score is not None]
        emotions = [msg.emotion for msg in messages if msg.emotion is not None]
        
        import numpy as np
        
        summary = {
            'total_messages': len(messages),
            'sentiment_distribution': {
                'positive': sentiments.count('positive'),
                'negative': sentiments.count('negative'),
                'neutral': sentiments.count('neutral')
            },
            'average_score': np.mean(scores) if scores else 0.0,
            'dominant_sentiment': max(set(sentiments), key=sentiments.count) if sentiments else 'neutral',
            'emotion_distribution': {emotion: emotions.count(emotion) for emotion in set(emotions)} if emotions else {}
        }
        
        return summary
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting sentiment summary: {str(e)}")

@router.get("/providers")
async def get_available_providers():
    """
    Get list of available LLM providers
    """
    try:
        return {
            'providers': ['openai', 'gemini', 'huggingface'],
            'default': settings.DEFAULT_LLM,
            'models': {
                'openai': settings.OPENAI_MODEL,
                'gemini': settings.GEMINI_MODEL,
                'huggingface': settings.HUGGINGFACE_MODEL
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting providers: {str(e)}")

@router.get("/tools")
async def get_available_tools():
    """
    Get list of available agent tools
    """
    try:
        tools = agent_system.get_available_tools()
        return {'tools': tools}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting tools: {str(e)}")

@router.post("/analyze-sentiment")
async def analyze_text_sentiment(text: str = Form(...)):
    """
    Analyze sentiment of provided text
    """
    try:
        result = await analyze_sentiment(text, use_hf=True)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

# Background task for sentiment analysis
async def analyze_user_message_sentiment(message: str, session_id: str):
    """
    Background task to analyze and store sentiment for user message
    """
    try:
        sentiment_result = await analyze_sentiment(message, use_hf=True)
        
        # Update the latest user message with sentiment data
        with get_db() as db:
            latest_msg = db.query(ChatMessage).filter(
                ChatMessage.session_id == session_id,
                ChatMessage.role == "user",
                ChatMessage.content == message
            ).order_by(ChatMessage.created_at.desc()).first()
            
            if latest_msg:
                latest_msg.sentiment_label = sentiment_result.get('label')
                latest_msg.sentiment_score = sentiment_result.get('score')
                latest_msg.emotion = sentiment_result.get('emotion')
                db.commit()
                
    except Exception as e:
        logger.error(f"Error in background sentiment analysis: {e}")
