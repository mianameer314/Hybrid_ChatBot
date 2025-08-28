"""
Advanced Streamlit Frontend for Agentic Chatbot
Features: File upload, web scraping, RAG, multi-LLM, sentiment analysis
"""
import streamlit as st
import requests
import json
import uuid
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="🤖 Advanced AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: auto;
    }
    
    .system-message {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        margin: 0 auto;
        text-align: center;
    }
    
    .sentiment-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        margin: 0.2rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .positive { background-color: #d4edda; color: #155724; }
    .negative { background-color: #f8d7da; color: #721c24; }
    .neutral { background-color: #e2e3e5; color: #383d41; }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "web_sources" not in st.session_state:
    st.session_state.web_sources = []

# Helper functions
def call_api(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None) -> Dict:
    """Call backend API"""
    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"
    
    try:
        if method == "POST":
            if files:
                response = requests.post(url, data=data, files=files)
            else:
                response = requests.post(url, json=data)
        elif method == "GET":
            response = requests.get(url, params=data)
        elif method == "DELETE":
            response = requests.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return {"error": str(e)}

def load_session_history(session_id: str):
    """Load chat history from backend"""
    try:
        response = call_api(f"/chat/sessions/{session_id}/messages")
        if "error" not in response:
            st.session_state.messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "sentiment_label": msg.get("sentiment_label"),
                    "emotion": msg.get("emotion"),
                    "created_at": msg["created_at"],
                    "model_used": msg.get("model_used")
                }
                for msg in response
            ]
    except Exception as e:
        st.error(f"Error loading history: {e}")

def display_message(message: Dict[str, Any]):
    """Display a chat message"""
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f'<div class="chat-message user-message">{content}</div>', unsafe_allow_html=True)
        
        # Show sentiment badges for user messages
        if message.get("sentiment_label"):
            sentiment_class = message["sentiment_label"]
            st.markdown(
                f'<span class="sentiment-badge {sentiment_class}">Sentiment: {sentiment_class.title()}</span>',
                unsafe_allow_html=True
            )
        
        if message.get("emotion"):
            st.markdown(
                f'<span class="sentiment-badge neutral">Emotion: {message["emotion"].title()}</span>',
                unsafe_allow_html=True
            )
            
    elif role == "assistant":
        st.markdown(f'<div class="chat-message assistant-message">{content}</div>', unsafe_allow_html=True)
        
        if message.get("model_used"):
            st.caption(f"🤖 Model: {message['model_used']}")
    
    elif role == "system":
        st.markdown(f'<div class="chat-message system-message">{content}</div>', unsafe_allow_html=True)

# Main app layout
col1, col2 = st.columns([3, 1])

with col2:
    st.header("⚙️ Settings")
    
    # Model selection
    st.subheader("🤖 AI Model")
    llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini", "huggingface"],
        index=0,
        help="Choose the AI model to use"
    )
    
    use_agent = st.checkbox(
        "Use Agent Mode",
        value=True,
        help="Enable advanced agent with tools and RAG capabilities"
    )
    
    # Session management
    st.subheader("💬 Session")
    
    # Session ID input
    new_session_id = st.text_input(
        "Session ID",
        value=st.session_state.session_id,
        help="Current chat session ID"
    )
    
    if new_session_id != st.session_state.session_id:
        st.session_state.session_id = new_session_id
        load_session_history(new_session_id)
        st.rerun()
    
    # New session button
    if st.button("🆕 New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Clear current session
    if st.button("🗑️ Clear Session"):
        result = call_api(f"/chat/sessions/{st.session_state.session_id}", method="DELETE")
        if "error" not in result:
            st.session_state.messages = []
            st.success("Session cleared!")
            st.rerun()
    
    # File upload section
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload documents for RAG",
        type=["pdf", "txt", "docx", "md"],
        help="Upload documents to add to the knowledge base"
    )
    
    if uploaded_file is not None and st.button("📤 Process Document"):
        with st.spinner("Processing document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"filename": uploaded_file.name}
                result = call_api("/documents/upload", method="POST", data=data, files=files)
                
                if "error" not in result:
                    st.success(f"Document processed successfully!")
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "id": result.get("document_id"),
                        "uploaded_at": datetime.now().isoformat()
                    })
                else:
                    st.error(f"Error: {result['error']}")
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")
    
    # Web scraping section
    st.subheader("🌐 Web Sources")
    web_url = st.text_input(
        "Add web source",
        placeholder="https://example.com",
        help="Add a web page to the knowledge base"
    )
    
    if web_url and st.button("🔗 Add Web Source"):
        with st.spinner("Scraping web content..."):
            try:
                result = call_api("/documents/web-scrape", method="POST", data={"urls": [web_url]})
                
                if "error" not in result:
                    st.success("Web content added successfully!")
                    st.session_state.web_sources.append({
                        "url": web_url,
                        "added_at": datetime.now().isoformat()
                    })
                else:
                    st.error(f"Error: {result['error']}")
            except Exception as e:
                st.error(f"Web scraping failed: {str(e)}")
    
    # Knowledge base section
    st.subheader("🧠 Knowledge Base")
    if st.button("📊 View RAG Stats"):
        with st.spinner("Loading statistics..."):
            try:
                stats = call_api("/rag/statistics")
                if "error" not in stats:
                    st.json(stats)
                else:
                    st.error(f"Error: {stats['error']}")
            except Exception as e:
                st.error(f"Failed to load stats: {str(e)}")

# Main chat interface
with col1:
    st.header("🤖 Advanced AI Chatbot")
    st.caption("Powered by RAG, Multi-LLM, and Sentiment Analysis")
    
    # Load session history on startup
    if not st.session_state.messages:
        load_session_history(st.session_state.session_id)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        if st.session_state.messages:
            for message in st.session_state.messages:
                display_message(message)
        else:
            st.info("👋 Hello! I'm your advanced AI assistant. I can help you with questions, analyze documents, search the web, and much more!")
    
    # Chat input
    if user_input := st.chat_input("Type your message..."):
        # Add user message to display
        user_message = {
            "role": "user",
            "content": user_input,
            "created_at": datetime.now().isoformat()
        }
        st.session_state.messages.append(user_message)
        
        # Display user message immediately
        with chat_container:
            display_message(user_message)
        
        # Generate bot response
        with st.spinner("🤔 Thinking..."):
            try:
                request_data = {
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "llm_provider": llm_provider,
                    "use_agent": use_agent
                }
                
                response = call_api("/chat/message", method="POST", data=request_data)
                
                if "error" not in response:
                    # Add assistant response
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "model_used": response["model_used"],
                        "processing_time": response["processing_time"],
                        "created_at": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(assistant_message)
                    
                    # Display assistant response
                    with chat_container:
                        display_message(assistant_message)
                    
                    # Show processing info
                    st.info(f"⚡ Response generated in {response['processing_time']:.2f}s using {response['model_used']}")
                    
                    if response.get("intermediate_steps"):
                        with st.expander("🔍 Agent Steps"):
                            for step in response["intermediate_steps"]:
                                st.write(step)
                
                else:
                    st.error(f"Error: {response['error']}")
                    
            except Exception as e:
                st.error(f"Failed to get response: {str(e)}")
        
        # Rerun to update chat
        st.rerun()

# Sidebar analytics
with st.sidebar:
    st.header("📈 Analytics")
    
    if st.button("📊 Session Analytics"):
        try:
            sentiment_summary = call_api(f"/chat/sessions/{st.session_state.session_id}/sentiment-summary")
            
            if "error" not in sentiment_summary and "total_messages" in sentiment_summary:
                st.subheader("😊 Sentiment Distribution")
                
                # Create pie chart for sentiment
                sentiment_data = sentiment_summary["sentiment_distribution"]
                if any(sentiment_data.values()):
                    fig_sentiment = px.pie(
                        values=list(sentiment_data.values()),
                        names=list(sentiment_data.keys()),
                        title="Message Sentiments",
                        color_discrete_map={
                            'positive': '#28a745',
                            'negative': '#dc3545', 
                            'neutral': '#6c757d'
                        }
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                
                # Emotion distribution
                if sentiment_summary.get("emotion_distribution"):
                    st.subheader("😌 Emotion Distribution")
                    emotion_data = sentiment_summary["emotion_distribution"]
                    
                    fig_emotion = px.bar(
                        x=list(emotion_data.keys()),
                        y=list(emotion_data.values()),
                        title="Detected Emotions"
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Messages", sentiment_summary["total_messages"])
                with col2:
                    st.metric("Avg Sentiment", f"{sentiment_summary.get('average_score', 0):.3f}")
                with col3:
                    st.metric("Dominant", sentiment_summary["dominant_sentiment"].title())
            
            else:
                st.info("No sentiment data available yet")
                
        except Exception as e:
            st.error(f"Analytics error: {e}")
    
    # System status
    st.subheader("🔧 System Status")
    if st.button("🏥 Health Check"):
        try:
            health = call_api("/health/status")
            if "error" not in health:
                st.success("✅ System Healthy")
                with st.expander("Details"):
                    st.json(health)
            else:
                st.error("❌ System Issues")
                st.json(health)
        except Exception as e:
            st.error(f"Health check failed: {e}")
    
    # RAG system info
    st.subheader("🎯 RAG System")
    if st.button("📚 View Knowledge Base"):
        try:
            stats = call_api("/rag/statistics")
            if "error" not in stats:
                st.write("📊 **Knowledge Base Statistics:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Documents", stats.get("documents", 0))
                    st.metric("Web Sources", stats.get("web_sources", 0))
                
                with col2:
                    st.metric("Total Vectors", stats.get("total_vectors", 0))
                    st.metric("KB Entries", stats.get("knowledge_base_entries", 0))
                
                st.caption(f"🔧 Model: {stats.get('embedding_model', 'Unknown')}")
            else:
                st.error(f"Error: {stats['error']}")
        except Exception as e:
            st.error(f"Failed to load RAG stats: {e}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🤖 Advanced AI Chatbot v2.0 | 
        Features: Multi-LLM • RAG • Sentiment Analysis • Agent Tools
    </div>
    """,
    unsafe_allow_html=True
)
