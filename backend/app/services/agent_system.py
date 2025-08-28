"""
Advanced LangChain Agent System with Tools and Memory
"""
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool, BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any, Optional, Type
import logging
import asyncio
from datetime import datetime
import json

from app.core.config import settings
from app.services.llm_providers import llm_manager
from app.services.rag_system import rag_system
from app.services.sentiment_analysis import analyze_sentiment
from app.core.database import get_db_session
from app.models import ChatSession, ChatMessage

logger = logging.getLogger(__name__)

# Agent prompt template
AGENT_PROMPT = """You are an advanced AI assistant with access to various tools and a comprehensive knowledge base. 
You can:

1. Search through documents, web content, and knowledge bases using RAG (Retrieval-Augmented Generation)
2. Analyze sentiment and emotions in user messages
3. Remember conversation history and context
4. Access structured knowledge and FAQ information
5. Process and understand uploaded documents and web content

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""

class RAGTool(BaseTool):
    """Tool for querying the RAG system"""
    
    name: str = "rag_search"
    description: str = "Search through documents, web content, and knowledge base for relevant information. Use this when you need to find specific information or answer questions based on uploaded content."
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        try:
            result = await rag_system.query(query)
            
            if result['retrieved_documents']:
                # Format the response with sources
                response_parts = [result['response']]
                response_parts.append("\nSources:")
                
                for doc in result['retrieved_documents']:
                    source_info = f"- {doc['source'].title()}"
                    if 'title' in doc['metadata']:
                        source_info += f": {doc['metadata']['title']}"
                    if 'url' in doc['metadata']:
                        source_info += f" ({doc['metadata']['url']})"
                    response_parts.append(source_info)
                
                return "\n".join(response_parts)
            else:
                return result['response']
                
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync implementation"""
        return asyncio.run(self._arun(query))

class SentimentTool(BaseTool):
    """Tool for sentiment analysis"""
    
    name: str = "sentiment_analysis"
    description: str = "Analyze the sentiment and emotion of text. Use this to understand the emotional tone of user messages or any text."
    
    async def _arun(self, text: str) -> str:
        """Async implementation"""
        try:
            result = await analyze_sentiment(text, use_hf=True)
            
            response = f"Sentiment Analysis Results:\n"
            response += f"- Sentiment: {result['label'].title()} (confidence: {result['confidence']:.3f})\n"
            
            if result.get('emotion'):
                response += f"- Emotion: {result['emotion'].title()} (confidence: {result['emotion_score']:.3f})\n"
            
            response += f"- Model used: {result['model']}"
            
            return response
            
        except Exception as e:
            return f"Error analyzing sentiment: {str(e)}"
    
    def _run(self, text: str) -> str:
        """Sync implementation"""
        return asyncio.run(self._arun(text))

class KnowledgeBaseTool(BaseTool):
    """Tool for accessing structured knowledge base"""
    
    name: str = "knowledge_base"
    description: str = "Access structured FAQ and knowledge base entries. Use this for company policies, procedures, common questions, and factual information."
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        try:
            # Search specifically in knowledge base
            with get_db_session() as db:
                from app.models import KnowledgeBase
                
                # Simple text search first
                kb_entries = db.query(KnowledgeBase).filter(
                    KnowledgeBase.question.contains(query) | 
                    KnowledgeBase.answer.contains(query)
                ).limit(3).all()
                
                if kb_entries:
                    responses = []
                    for entry in kb_entries:
                        responses.append(f"Q: {entry.question}\nA: {entry.answer}")
                        if entry.category:
                            responses.append(f"Category: {entry.category}")
                        responses.append("---")
                    
                    return "\n".join(responses)
                else:
                    # Fallback to RAG search
                    rag_result = await rag_system.query(query, k=3)
                    return rag_result['response']
                    
        except Exception as e:
            return f"Error accessing knowledge base: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync implementation"""
        return asyncio.run(self._arun(query))

class WebSearchTool(BaseTool):
    """Tool for web search and content retrieval"""
    
    name: str = "web_search"
    description: str = "Search and retrieve information from web sources that have been added to the knowledge base."
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        try:
            # Search web content in vector store
            result = await rag_system.query(query, k=3)
            
            # Filter for web sources only
            web_docs = [doc for doc in result['retrieved_documents'] if doc['source'] == 'web']
            
            if web_docs:
                response_parts = [f"Found {len(web_docs)} relevant web sources:\n"]
                
                for doc in web_docs:
                    response_parts.append(f"- Source: {doc['metadata'].get('title', 'Untitled')}")
                    if 'url' in doc['metadata']:
                        response_parts.append(f"  URL: {doc['metadata']['url']}")
                    response_parts.append(f"  Content: {doc['content']}")
                    response_parts.append("")
                
                return "\n".join(response_parts)
            else:
                return "No relevant web content found for this query."
                
        except Exception as e:
            return f"Error searching web content: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync implementation"""
        return asyncio.run(self._arun(query))

class DocumentSearchTool(BaseTool):
    """Tool for searching uploaded documents"""
    
    name: str = "document_search"
    description: str = "Search through uploaded documents (PDFs, Word docs, text files) for relevant information."
    
    async def _arun(self, query: str) -> str:
        """Async implementation"""
        try:
            # Search documents in vector store
            result = await rag_system.query(query, k=3)
            
            # Filter for document sources only
            doc_results = [doc for doc in result['retrieved_documents'] if doc['source'] == 'document']
            
            if doc_results:
                response_parts = [f"Found {len(doc_results)} relevant documents:\n"]
                
                for doc in doc_results:
                    response_parts.append(f"- Document ID: {doc['metadata'].get('document_id', 'Unknown')}")
                    response_parts.append(f"  Content: {doc['content']}")
                    response_parts.append(f"  Relevance: {doc['score']:.3f}")
                    response_parts.append("")
                
                return "\n".join(response_parts)
            else:
                return "No relevant documents found for this query."
                
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    def _run(self, query: str) -> str:
        """Sync implementation"""
        return asyncio.run(self._arun(query))

class DatabaseMemory:
    """Custom memory that persists to database"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self._load_history()
    
    def _load_history(self):
        """Load conversation history from database"""
        try:
            with get_db_session() as db:
                db_messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == self.session_id
                ).order_by(ChatMessage.created_at.asc()).limit(20).all()  # Last 20 messages
                
                self.messages = []
                for msg in db_messages:
                    if msg.role == "user":
                        self.messages.append(HumanMessage(content=msg.content))
                    elif msg.role == "assistant":
                        self.messages.append(AIMessage(content=msg.content))
                    elif msg.role == "system":
                        self.messages.append(SystemMessage(content=msg.content))
                
        except Exception as e:
            logger.error(f"Error loading chat history: {e}")
            self.messages = []
    
    def add_message(self, message: BaseMessage):
        """Add a message to memory"""
        self.messages.append(message)
        
        # Keep only last 10 messages in memory
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages"""
        return self.messages
    
    def clear(self):
        """Clear memory"""
        self.messages = []

class AgentSystem:
    """Advanced LangChain agent system"""
    
    def __init__(self):
        self.agents = {}  # Cache agents by session
        self.tools = self._create_tools()
        self.prompt = PromptTemplate.from_template(AGENT_PROMPT)
    
    def _create_tools(self) -> List[BaseTool]:
        """Create agent tools"""
        return [
            RAGTool(),
            SentimentTool(),
            KnowledgeBaseTool(),
            WebSearchTool(),
            DocumentSearchTool()
        ]
    
    async def get_agent(self, session_id: str, llm_provider: str = None) -> AgentExecutor:
        """Get or create an agent for a session"""
        agent_key = f"{session_id}_{llm_provider or settings.DEFAULT_LLM}"
        
        if agent_key not in self.agents:
            try:
                # Get LLM
                provider = await llm_manager.get_provider(llm_provider)
                
                # Create LangChain LLM wrapper
                from langchain_core.language_models.llms import LLM
                from langchain_core.outputs import LLMResult, Generation
                from langchain_core.callbacks.manager import CallbackManagerForLLMRun
                from pydantic import Field
                
                class LLMWrapper(LLM):
                    provider: Any = Field(exclude=True)
                    
                    def __init__(self, provider, **kwargs):
                        super().__init__(**kwargs)
                        self.provider = provider
                    
                    @property
                    def _llm_type(self) -> str:
                        return "custom_wrapper"
                    
                    def _call(
                        self,
                        prompt: str,
                        stop: Optional[List[str]] = None,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        **kwargs: Any,
                    ) -> str:
                        """Call the LLM with a prompt and return the response."""
                        try:
                            # Convert prompt to messages format for chat completion
                            messages = [{"role": "user", "content": prompt}]
                            response = asyncio.run(self.provider.chat_completion(messages, **kwargs))
                            return response
                        except Exception as e:
                            logger.error(f"LLM call error: {e}")
                            return f"Error: {str(e)}"
                    
                    async def _acall(
                        self,
                        prompt: str,
                        stop: Optional[List[str]] = None,
                        run_manager: Optional[CallbackManagerForLLMRun] = None,
                        **kwargs: Any,
                    ) -> str:
                        """Async call the LLM with a prompt and return the response."""
                        try:
                            # Convert prompt to messages format for chat completion
                            messages = [{"role": "user", "content": prompt}]
                            response = await self.provider.chat_completion(messages, **kwargs)
                            return response
                        except Exception as e:
                            logger.error(f"Async LLM call error: {e}")
                            return f"Error: {str(e)}"
                
                llm_wrapper = LLMWrapper(provider)
                
                # Create memory
                memory = DatabaseMemory(session_id)
                
                # Create agent
                agent = create_react_agent(
                    llm=llm_wrapper,
                    tools=self.tools,
                    prompt=self.prompt
                )
                
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    memory=ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    ),
                    verbose=settings.DEBUG,
                    handle_parsing_errors=True,
                    max_iterations=3,
                    max_execution_time=60
                )
                
                self.agents[agent_key] = agent_executor
                logger.info(f"Created agent for session {session_id}")
                
            except Exception as e:
                logger.error(f"Error creating agent: {e}")
                raise
        
        return self.agents[agent_key]
    
    async def process_message(
        self, 
        session_id: str, 
        message: str, 
        llm_provider: str = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a message through the agent system"""
        start_time = datetime.now()
        
        try:
            # Get agent
            agent = await self.get_agent(session_id, llm_provider)
            
            # Add context if provided
            input_data = {"input": message}
            if context:
                input_data["context"] = json.dumps(context)
            
            # Run agent
            result = await asyncio.to_thread(agent.invoke, input_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save to database
            await self._save_interaction(
                session_id=session_id,
                user_message=message,
                assistant_response=result.get('output', ''),
                llm_provider=llm_provider or settings.DEFAULT_LLM,
                processing_time=processing_time,
                metadata=context
            )
            
            return {
                'response': result.get('output', ''),
                'intermediate_steps': result.get('intermediate_steps', []),
                'processing_time': processing_time,
                'session_id': session_id,
                'model_used': llm_provider or settings.DEFAULT_LLM
            }
            
        except Exception as e:
            logger.error(f"Agent processing error: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'response': f"I encountered an error while processing your request: {str(e)}",
                'intermediate_steps': [],
                'processing_time': processing_time,
                'session_id': session_id,
                'error': str(e)
            }
    
    async def _save_interaction(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        llm_provider: str,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save interaction to database"""
        try:
            with get_db_session() as db:
                # Ensure session exists
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if not session:
                    session = ChatSession(
                        id=session_id,
                        title=user_message[:50] + "..." if len(user_message) > 50 else user_message
                    )
                    db.add(session)
                    db.flush()
                
                # Analyze sentiment for user message
                sentiment_result = await analyze_sentiment(user_message, use_hf=True)
                
                # Save user message
                user_msg = ChatMessage(
                    session_id=session_id,
                    role="user",
                    content=user_message,
                    sentiment_label=sentiment_result.get('label'),
                    sentiment_score=sentiment_result.get('score'),
                    emotion=sentiment_result.get('emotion'),
                    model_used=llm_provider,
                    processing_time=processing_time,
                    extra_metadata=metadata
                )
                db.add(user_msg)
                
                # Save assistant response
                assistant_msg = ChatMessage(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_response,
                    model_used=llm_provider,
                    processing_time=processing_time,
                    extra_metadata=metadata
                )
                db.add(assistant_msg)
                
                db.commit()
                
        except Exception as e:
            logger.error(f"Error saving interaction: {e}")
    
    async def get_session_history(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        try:
            with get_db_session() as db:
                messages = db.query(ChatMessage).filter(
                    ChatMessage.session_id == session_id
                ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
                
                return [
                    {
                        'role': msg.role,
                        'content': msg.content,
                        'sentiment_label': msg.sentiment_label,
                        'emotion': msg.emotion,
                        'created_at': msg.created_at.isoformat(),
                        'model_used': msg.model_used
                    }
                    for msg in reversed(messages)  # Reverse to get chronological order
                ]
                
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    async def clear_session(self, session_id: str) -> bool:
        """Clear a session's conversation history"""
        try:
            with get_db_session() as db:
                # Delete messages
                db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
                
                # Update session
                session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
                if session:
                    session.is_active = False
                
                db.commit()
                
                # Clear agent cache
                agent_keys = [key for key in self.agents.keys() if key.startswith(session_id)]
                for key in agent_keys:
                    del self.agents[key]
                
                logger.info(f"Cleared session {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session: {e}")
            return False
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools"""
        return [
            {
                'name': tool.name,
                'description': tool.description
            }
            for tool in self.tools
        ]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get agent system status"""
        try:
            # RAG system stats
            rag_stats = await rag_system.get_statistics()
            
            # Active sessions count
            with get_db_session() as db:
                active_sessions = db.query(ChatSession).filter(ChatSession.is_active == True).count()
                total_messages = db.query(ChatMessage).count()
            
            return {
                'status': 'healthy',
                'active_agents': len(self.agents),
                'available_tools': len(self.tools),
                'active_sessions': active_sessions,
                'total_messages': total_messages,
                'rag_system': rag_stats,
                'default_llm': settings.DEFAULT_LLM,
                'embedding_model': settings.EMBEDDING_MODEL
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'status': 'error', 'error': str(e)}

# Global agent system instance
agent_system = AgentSystem()
