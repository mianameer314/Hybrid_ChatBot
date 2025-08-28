# Advanced AI Chatbot System - Complete Guide
*From Zero to Hero: Understanding Every Concept and Implementation*

---

## Table of Contents
1. [The Story Behind Our Project](#the-story-behind-our-project)
2. [Architecture Overview](#architecture-overview)  
3. [Core Technologies & Concepts](#core-technologies--concepts)
4. [Step-by-Step Implementation Journey](#step-by-step-implementation-journey)
5. [Database Design & Models](#database-design--models)
6. [RAG System Deep Dive](#rag-system-deep-dive)
7. [Agent System Architecture](#agent-system-architecture)
8. [LLM Provider System](#llm-provider-system)
9. [Frontend Implementation](#frontend-implementation)
10. [Security & Performance](#security--performance)
11. [Deployment & Production](#deployment--production)
12. [Troubleshooting Guide](#troubleshooting-guide)

---

## The Story Behind Our Project

### The Vision
Imagine you're a developer who wants to build the most advanced chatbot possible. Not just any chatbot, but one that can:
- **Remember** every conversation
- **Learn** from uploaded documents
- **Search** the web for information
- **Understand** emotions and sentiment
- **Switch** between different AI models
- **Process** PDF files, Word documents, and more
- **Analyze** user behavior and patterns

This is exactly what we built - a **Hybrid AI Chatbot System** that combines the best of multiple worlds.

### The Journey
Our journey began with a simple question: *"Why should users be limited to just one AI model or one type of functionality?"*

We decided to create a system that:
1. **Integrates multiple AI providers** (OpenAI, Gemini, Hugging Face)
2. **Implements RAG (Retrieval-Augmented Generation)** for document understanding
3. **Uses LangChain agents** for intelligent tool selection
4. **Maintains conversation memory** in a database
5. **Provides real-time sentiment analysis**
6. **Offers a beautiful, responsive frontend**

---

## Architecture Overview

### The Big Picture
Think of our system as a **smart orchestra conductor** that coordinates different musicians (AI models, databases, tools) to create beautiful music (intelligent conversations).

```
Frontend (streamlit)
    ↓
API Gateway (FastAPI)
    ↓
┌─────────────────────────────────────────┐
│           Core Engine                   │
├─────────────────────────────────────────┤
│  Agent System (LangChain)              │
│  ├── RAG Tool                          │
│  ├── Document Search Tool              │
│  ├── Sentiment Analysis Tool           │
│  ├── Knowledge Base Tool               │
│  └── Web Search Tool                   │
├─────────────────────────────────────────┤
│  LLM Provider Manager                   │
│  ├── OpenAI GPT                        │
│  ├── Google Gemini                     │
│  └── Hugging Face Models               │
├─────────────────────────────────────────┤
│  RAG System                            │
│  ├── Document Processor                │
│  ├── Embedding Manager                 │
│  ├── Vector Store (FAISS)              │
│  └── Web Scraper                       │
└─────────────────────────────────────────┘
    ↓
Database Layer (PostgreSQL)
    ↓
External Services (Redis, File Storage)
```

### Key Components Explained

#### 1. **Frontend Layer**
- **Technology**: streamlit with modern UI components
- **Purpose**: Beautiful, responsive interface for users
- **Features**: Real-time chat, file uploads, settings, analytics

#### 2. **API Layer** 
- **Technology**: FastAPI (Python)
- **Purpose**: High-performance REST API
- **Features**: Async processing, automatic documentation, validation

#### 3. **Agent System**
- **Technology**: LangChain
- **Purpose**: Intelligent decision-making
- **Features**: Tool selection, reasoning, memory management

#### 4. **RAG System**
- **Technology**: Sentence Transformers + FAISS
- **Purpose**: Document understanding and retrieval
- **Features**: PDF processing, vector search, web scraping

#### 5. **Database Layer**
- **Technology**: PostgreSQL + SQLAlchemy
- **Purpose**: Data persistence and relationships
- **Features**: Chat history, user management, document storage

---

## Core Technologies & Concepts

### 1. RAG (Retrieval-Augmented Generation)
**What is RAG?**
Imagine you're taking an open-book exam. Instead of memorizing everything, you can look up information from books when needed. RAG works similarly for AI:

1. **Store Information**: We store documents, web pages, and knowledge in a searchable format
2. **Retrieve Relevant Info**: When a user asks a question, we find relevant information
3. **Generate Answer**: We combine the retrieved information with the AI's knowledge to generate accurate answers

**Our RAG Implementation:**
```python
# Step 1: Process Documents
document → text extraction → chunks → embeddings → vector store

# Step 2: User Query
user question → embedding → vector search → relevant chunks

# Step 3: Generate Response
relevant chunks + user question → LLM → enhanced answer
```

### 2. LangChain Agents
**What are Agents?**
Think of an agent as a **smart assistant** that can use different tools to solve problems. Just like a human would:
1. **Think**: Analyze the problem
2. **Choose Tools**: Pick the right tool for the job
3. **Act**: Use the tool
4. **Observe**: See the results
5. **Repeat**: Until the problem is solved

**Our Agent Tools:**
- **RAG Tool**: Search documents and knowledge base
- **Document Search**: Find uploaded PDFs and files
- **Sentiment Analysis**: Understand user emotions
- **Web Search**: Find information from scraped websites
- **Knowledge Base**: Access structured Q&A data

### 3. Vector Embeddings
**What are Embeddings?**
Imagine converting every word, sentence, or document into a **unique fingerprint** (a list of numbers). Similar content has similar fingerprints.

```python
"Hello world" → [0.1, 0.8, 0.3, -0.5, ...]
"Hi there"    → [0.2, 0.7, 0.4, -0.4, ...]  # Similar numbers!
"Cat food"    → [0.9, -0.2, 0.1, 0.8, ...]  # Different numbers!
```

**Why This Matters:**
- We can find **similar content** by comparing these numbers
- **Fast searching** through millions of documents
- **Semantic understanding** (meaning-based, not just keyword matching)

### 4. Multi-LLM Provider System
**The Problem**: What if OpenAI is down? What if Gemini is better for certain tasks?

**Our Solution**: A **provider manager** that can:
- **Switch between models** automatically
- **Handle failures gracefully** with fallbacks  
- **Balance costs** by using cheaper models when appropriate
- **Optimize performance** by choosing the fastest model

---

## Step-by-Step Implementation Journey

### Phase 1: Foundation Setup (Day 1-2)

#### Step 1: Project Structure
```
backend/
├── app/
│   ├── api/          # REST API endpoints
│   ├── core/         # Configuration and utilities
│   ├── models/       # Database models
│   ├── services/     # Business logic
│   └── main.py       # Application entry point
└── requirements.txt  # Dependencies
```

#### Step 2: Database Models
We started with the most important data structures:

**Chat Sessions**: Every conversation is a session
```python
class ChatSession(Base):
    id = Column(String, primary_key=True)
    title = Column(String)
    created_at = Column(DateTime)
    is_active = Column(Boolean)
```

**Chat Messages**: Individual messages in conversations
```python
class ChatMessage(Base):
    id = Column(String, primary_key=True) 
    session_id = Column(String, ForeignKey('chat_sessions.id'))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    sentiment_label = Column(String)
    emotion = Column(String)
    created_at = Column(DateTime)
```

#### Step 3: Basic API Setup
```python
# main.py - The heart of our API
from fastapi import FastAPI

app = FastAPI(
    title="Advanced AI Chatbot",
    description="Multi-LLM chatbot with RAG capabilities"
)

@app.post("/chat")
async def chat_endpoint(message: ChatRequest):
    # This is where the magic happens!
    pass
```

### Phase 2: LLM Provider System (Day 3-4)

#### The Challenge
How do we support multiple AI providers (OpenAI, Gemini, Hugging Face) with different APIs?

#### Our Solution: Provider Pattern
```python
class BaseLLMProvider:
    async def chat_completion(self, messages: List[Dict]) -> str:
        raise NotImplementedError

class OpenAIProvider(BaseLLMProvider):
    async def chat_completion(self, messages: List[Dict]) -> str:
        # OpenAI-specific implementation
        
class GeminiProvider(BaseLLMProvider):
    async def chat_completion(self, messages: List[Dict]) -> str:
        # Gemini-specific implementation
```

#### Provider Manager: The Orchestrator
```python
class LLMProviderManager:
    def __init__(self):
        self.providers = {
            'openai': OpenAIProvider(),
            'gemini': GeminiProvider(),
            'huggingface': HuggingFaceProvider()
        }
    
    async def generate_response_with_fallback(self, messages, provider_name=None):
        # Try primary provider
        # If it fails, try backup providers
        # Return error only if all fail
```

### Phase 3: Document Processing & RAG (Day 5-7)

#### The Challenge
How do we make the chatbot understand and search through uploaded documents?

#### Step 1: Document Processing
```python
class DocumentProcessor:
    async def process_pdf(self, file_path: str):
        # Extract text from PDF
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
        
        return {
            'content': '\n'.join(text),
            'metadata': {...}
        }
```

#### Step 2: Text Chunking
**Why Chunking?**: LLMs have token limits. Long documents must be split into smaller pieces.

```python
def create_chunks(self, text: str) -> List[str]:
    # Split into overlapping chunks
    # Chunk size: 1000 characters
    # Overlap: 200 characters (to maintain context)
    
    chunks = []
    for i in range(0, len(text), 800):  # 1000-200 overlap
        chunk = text[i:i+1000]
        chunks.append(chunk)
    return chunks
```

#### Step 3: Embeddings Generation
```python
class EmbeddingManager:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def encode_text(self, text: str) -> List[float]:
        # Convert text to 384-dimensional vector
        embedding = await asyncio.to_thread(self.model.encode, [text])
        return embedding[0].tolist()
```

#### Step 4: Vector Storage with FAISS
```python
class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(384)  # 384 dimensions
        self.documents = []
    
    def add_vectors(self, embeddings: List[List[float]], docs: List[Dict]):
        # Normalize for cosine similarity
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        # Add to FAISS index
        self.index.add(embeddings_np)
        self.documents.extend(docs)
    
    def search(self, query_embedding: List[float], k: int = 5):
        # Find k most similar documents
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        
        scores, indices = self.index.search(query_np, k)
        return [(self.documents[idx], score) for idx, score in zip(indices[0], scores[0])]
```

### Phase 4: Agent System Implementation (Day 8-10)

#### The Challenge
How do we make the chatbot intelligently choose which tools to use?

#### Our LangChain Agent Architecture
```python
# Agent Tools
class DocumentSearchTool(BaseTool):
    name = "document_search" 
    description = "Search uploaded documents for information"
    
    async def _arun(self, query: str) -> str:
        # Search database for documents
        # Return relevant content
        
class RAGTool(BaseTool):
    name = "rag_search"
    description = "Search knowledge base and documents"
    
    async def _arun(self, query: str) -> str:
        # Query vector store
        # Generate contextual response
```

#### Agent Prompt Engineering
```python
AGENT_PROMPT = """You are an advanced AI assistant with access to tools.

Available tools:
{tools}

When users mention "CV", "resume", "pdf", always use document_search first.

Use this format:
Question: the input question
Thought: analyze what to do  
Action: choose a tool from [{tool_names}]
Action Input: input for the tool
Observation: tool result
... (repeat as needed)
Final Answer: your final response
"""
```

#### The Agent in Action
When you ask "summarize my CV PDF":

1. **Thought**: User wants CV summary, I should search documents
2. **Action**: document_search  
3. **Action Input**: "CV PDF resume"
4. **Observation**: Found CV content...
5. **Thought**: Now I can summarize this
6. **Final Answer**: Here's your CV summary: ...

### Phase 5: Sentiment Analysis (Day 11)

#### Why Sentiment Analysis?
Understanding user emotions helps us:
- **Adjust response tone** (empathetic for frustrated users)
- **Track user satisfaction** 
- **Provide analytics** to improve the system

#### Implementation
```python
async def analyze_sentiment(text: str, use_hf: bool = True):
    if use_hf:
        # Use Hugging Face transformer model
        model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        
        # Convert to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'label': 'positive'|'negative'|'neutral',
            'confidence': float(max_prob),
            'model': 'roberta-sentiment'
        }
```

### Phase 6: Web Scraping Integration (Day 12)

#### The Feature
Allow the chatbot to learn from websites!

#### Implementation
```python
class WebScraper:
    async def scrape_url(self, url: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove scripts and styles
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract clean text
                text = soup.get_text()
                title = soup.find('title').get_text()
                
                return {
                    'url': url,
                    'title': title, 
                    'content': clean_text,
                    'scraped_at': datetime.utcnow()
                }
```

### Phase 7: Frontend Development (Day 13-15)

#### Technology Stack
- **React** with **Next.js** for SSR capabilities
- **TypeScript** for type safety
- **Tailwind CSS** for styling
- **WebSocket** for real-time chat

#### Key Components

**Chat Interface**:
```tsx
const ChatInterface = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    
    const sendMessage = async () => {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: input,
                session_id: sessionId
            })
        });
        
        const data = await response.json();
        setMessages(prev => [...prev, data]);
    };
    
    return (
        <div className="chat-container">
            <MessageList messages={messages} />
            <ChatInput onSend={sendMessage} />
        </div>
    );
};
```

**File Upload Component**:
```tsx
const FileUpload = () => {
    const handleFileUpload = async (file: File) => {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            toast.success('File uploaded successfully!');
        }
    };
    
    return (
        <Dropzone onDrop={handleFileUpload}>
            <p>Drop files here or click to upload</p>
        </Dropzone>
    );
};
```

### Phase 8: Advanced Features (Day 16-20)

#### 1. Session Management
```python
class SessionManager:
    def __init__(self):
        self.active_sessions = {}
    
    async def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        session = ChatSession(
            id=session_id,
            title="New Conversation",
            created_at=datetime.utcnow(),
            is_active=True
        )
        # Save to database
        return session_id
    
    async def get_session_history(self, session_id: str):
        # Retrieve chat history from database
        pass
```

#### 2. Caching System
```python
import redis

cache = redis.Redis(host='localhost', port=6379, db=0)

async def cached_embedding(text: str) -> List[float]:
    cache_key = f"embedding:{hash(text)}"
    
    # Try to get from cache
    cached = cache.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Generate new embedding
    embedding = await embedding_manager.encode_text(text)
    
    # Cache for 1 hour
    cache.setex(cache_key, 3600, json.dumps(embedding))
    
    return embedding
```

#### 3. Rate Limiting
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("10/minute")  # Max 10 requests per minute
async def chat_endpoint(request: Request, message: ChatRequest):
    # Process chat message
    pass
```

---

## Database Design & Models

### Core Database Schema

#### 1. Chat Sessions Table
```sql
CREATE TABLE chat_sessions (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    user_id VARCHAR(36),  -- Future: user management
    metadata JSONB        -- Flexible additional data
);
```

#### 2. Chat Messages Table
```sql
CREATE TABLE chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) REFERENCES chat_sessions(id),
    role VARCHAR(20) NOT NULL, -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Sentiment Analysis
    sentiment_label VARCHAR(20),  -- 'positive', 'negative', 'neutral'
    sentiment_score FLOAT,
    emotion VARCHAR(50),
    
    -- Model Info
    model_used VARCHAR(50),
    processing_time FLOAT,
    
    -- Metadata
    extra_metadata JSONB
);
```

#### 3. Documents Table
```sql
CREATE TABLE documents (
    id VARCHAR(36) PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_name VARCHAR(255) NOT NULL,
    file_type VARCHAR(10) NOT NULL, -- 'pdf', 'docx', 'txt'
    file_size BIGINT NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    
    -- Content
    content TEXT,
    title VARCHAR(255),
    author VARCHAR(255),
    summary TEXT,
    
    -- Processing
    processing_status VARCHAR(20) DEFAULT 'pending',
    processed_at TIMESTAMP,
    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    extra_metadata JSONB
);
```

#### 4. Document Chunks Table
```sql
CREATE TABLE document_chunks (
    id VARCHAR(36) PRIMARY KEY,
    document_id VARCHAR(36) REFERENCES documents(id),
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    
    -- Vector Embedding (stored as JSON array)
    embedding JSONB,
    embedding_model VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 5. Knowledge Base Table
```sql
CREATE TABLE knowledge_base (
    id VARCHAR(36) PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    category VARCHAR(100),
    tags TEXT[], -- Array of tags
    
    -- Vector Embedding
    embedding JSONB,
    embedding_model VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 6. Web Sources Table
```sql
CREATE TABLE web_sources (
    id VARCHAR(36) PRIMARY KEY,
    url VARCHAR(1000) NOT NULL UNIQUE,
    domain VARCHAR(255),
    title VARCHAR(500),
    content TEXT,
    
    -- Scraping Info
    scraped_at TIMESTAMP,
    status_code INTEGER,
    content_type VARCHAR(100),
    
    -- SEO Data
    meta_description TEXT,
    keywords TEXT[],
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### 7. Web Chunks Table
```sql
CREATE TABLE web_chunks (
    id VARCHAR(36) PRIMARY KEY,
    source_id VARCHAR(36) REFERENCES web_sources(id),
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    
    -- Vector Embedding
    embedding JSONB,
    embedding_model VARCHAR(100),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Database Relationships

```
chat_sessions (1) ←→ (many) chat_messages
documents (1) ←→ (many) document_chunks  
web_sources (1) ←→ (many) web_chunks
```

### Indexing Strategy

```sql
-- Performance indexes
CREATE INDEX idx_chat_messages_session_id ON chat_messages(session_id);
CREATE INDEX idx_chat_messages_created_at ON chat_messages(created_at);
CREATE INDEX idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX idx_web_chunks_source_id ON web_chunks(source_id);

-- Text search indexes
CREATE INDEX idx_documents_content ON documents USING GIN(to_tsvector('english', content));
CREATE INDEX idx_knowledge_base_question ON knowledge_base USING GIN(to_tsvector('english', question));
```

---

## RAG System Deep Dive

### What Makes Our RAG System Special?

Our RAG system is like a **smart librarian** that can:
1. **Organize** any type of document
2. **Find** relevant information instantly  
3. **Understand** context and meaning
4. **Combine** information from multiple sources

### Architecture Deep Dive

#### 1. Document Processing Pipeline
```
Raw Document → Text Extraction → Preprocessing → Chunking → Embeddings → Vector Store
     ↓              ↓              ↓           ↓          ↓           ↓
   PDF/DOCX    Clean Text    Remove Noise   Chunks    Vector     FAISS Index
```

#### 2. Embedding Strategy
**Why Sentence Transformers?**
- **Multilingual** support
- **Semantic** understanding (meaning-based)
- **Fast** inference
- **Good quality** for general domains

```python
# Our embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# Produces 384-dimensional vectors
# Trained on 1B+ sentence pairs
```

#### 3. Retrieval Process
```python
async def retrieve_relevant_docs(query: str, k: int = 5):
    # Step 1: Convert query to embedding
    query_embedding = await embedding_manager.encode_text(query)
    
    # Step 2: Search vector store
    candidates = vector_store.search(query_embedding, k=k*2)
    
    # Step 3: Rerank by relevance score
    filtered = [(doc, score) for doc, score in candidates 
                if score >= SIMILARITY_THRESHOLD]
    
    # Step 4: Return top k results
    return filtered[:k]
```

#### 4. Context Assembly
```python
def create_context(retrieved_docs: List[Tuple[Dict, float]]) -> str:
    context_parts = []
    
    for doc, score in retrieved_docs:
        context_parts.append(f"Source: {doc['source']}")
        context_parts.append(f"Content: {doc['content']}")
        context_parts.append(f"Relevance: {score:.3f}")
        context_parts.append("---")
    
    return "\n".join(context_parts)
```

#### 5. Response Generation
```python
async def generate_rag_response(query: str, context: str) -> str:
    system_prompt = """
    You are a helpful assistant. Use the provided context to answer questions.
    If the context doesn't contain relevant information, say so clearly.
    Always cite your sources when possible.
    """
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
    
    response = await llm_manager.generate_response(messages)
    return response
```

### Advanced RAG Techniques

#### 1. Hybrid Search
**Problem**: Pure vector search isn't always perfect.
**Solution**: Combine vector search with traditional keyword search.

```python
async def hybrid_search(query: str, k: int = 5):
    # Vector search
    vector_results = await vector_search(query, k)
    
    # Keyword search (PostgreSQL full-text search)
    keyword_results = await keyword_search(query, k)
    
    # Combine and rerank results
    combined = merge_and_rerank(vector_results, keyword_results)
    
    return combined[:k]
```

#### 2. Context Window Management
**Problem**: LLMs have token limits.
**Solution**: Smart context truncation.

```python
def fit_context_to_window(context: str, max_tokens: int = 3000) -> str:
    # Estimate tokens (rough: 4 chars = 1 token)
    estimated_tokens = len(context) // 4
    
    if estimated_tokens <= max_tokens:
        return context
    
    # Truncate but try to keep complete sentences
    target_length = max_tokens * 4
    sentences = context.split('.')
    
    truncated = ""
    for sentence in sentences:
        if len(truncated + sentence) > target_length:
            break
        truncated += sentence + "."
    
    return truncated
```

#### 3. Multi-Document Synthesis
**Problem**: Answer might need information from multiple documents.
**Solution**: Multi-step retrieval and synthesis.

```python
async def multi_doc_synthesis(query: str) -> str:
    # Step 1: Initial retrieval
    initial_docs = await retrieve_relevant_docs(query, k=3)
    
    # Step 2: Generate sub-questions
    sub_questions = await generate_sub_questions(query, initial_docs)
    
    # Step 3: Retrieve for each sub-question
    all_docs = []
    for sub_q in sub_questions:
        docs = await retrieve_relevant_docs(sub_q, k=2)
        all_docs.extend(docs)
    
    # Step 4: Remove duplicates and synthesize
    unique_docs = remove_duplicates(all_docs)
    final_context = create_context(unique_docs)
    
    return await generate_rag_response(query, final_context)
```

---

## Agent System Architecture

### The Philosophy Behind Agents

Traditional chatbots are like **calculators** - they can only do one thing at a time. Our agent is like a **Swiss Army knife** - it has many tools and knows which one to use for each situation.

### Agent Decision-Making Process

#### 1. The ReAct Framework
**ReAct** = **Reasoning** + **Acting**

```
Question: "Summarize my CV PDF"
↓
Thought: "User wants CV summary. I should search for uploaded documents."
↓  
Action: document_search
↓
Action Input: "CV PDF resume"
↓
Observation: "Found 2 PDF documents that could be your CV: ..."
↓
Thought: "Great! Now I have the CV content. I should provide a summary."
↓
Final Answer: "Here's a summary of your CV: ..."
```

#### 2. Tool Selection Logic
```python
class ToolSelector:
    def __init__(self):
        self.tool_keywords = {
            'document_search': ['cv', 'resume', 'pdf', 'document', 'file'],
            'sentiment_analysis': ['feel', 'emotion', 'mood', 'sentiment'],
            'rag_search': ['search', 'find', 'information', 'about'],
            'knowledge_base': ['policy', 'procedure', 'faq', 'guideline']
        }
    
    def suggest_tool(self, query: str) -> str:
        query_lower = query.lower()
        
        scores = {}
        for tool, keywords in self.tool_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            scores[tool] = score
        
        return max(scores, key=scores.get)
```

#### 3. Memory Management
**Problem**: Agents can forget context between iterations.
**Solution**: Persistent memory system.

```python
class AgentMemory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.short_term = []  # Current conversation
        self.long_term = []   # Historical context
        self.working = {}     # Temporary variables
    
    def add_interaction(self, thought: str, action: str, result: str):
        interaction = {
            'thought': thought,
            'action': action, 
            'result': result,
            'timestamp': datetime.utcnow()
        }
        self.short_term.append(interaction)
        
        # Keep only last 5 interactions in short-term memory
        if len(self.short_term) > 5:
            self.short_term.pop(0)
    
    def get_context(self) -> str:
        if not self.short_term:
            return ""
        
        context_parts = ["Recent interactions:"]
        for interaction in self.short_term[-3:]:  # Last 3 interactions
            context_parts.append(f"Action: {interaction['action']}")
            context_parts.append(f"Result: {interaction['result'][:100]}...")
        
        return "\n".join(context_parts)
```

### Advanced Agent Features

#### 1. Error Recovery
**What happens when tools fail?**

```python
class ResilientAgent:
    async def execute_action(self, action: str, input_data: str):
        try:
            tool = self.get_tool(action)
            result = await tool.run(input_data)
            return result
        except Exception as e:
            # Log the error
            logger.error(f"Tool {action} failed: {e}")
            
            # Try alternative approach
            if action == "rag_search":
                # Fallback to simple database search
                return await self.simple_database_search(input_data)
            
            elif action == "document_search":
                # Fallback to listing all documents
                return await self.list_all_documents()
            
            # Generic fallback
            return f"I encountered an issue with {action}. Let me try a different approach."
```

#### 2. Performance Optimization
**Challenge**: Agents can be slow with multiple iterations.
**Solution**: Smart early stopping and parallel processing.

```python
class OptimizedAgent:
    def __init__(self):
        self.max_iterations = 2  # Reduced from default 3
        self.max_execution_time = 30  # 30 seconds max
        self.confidence_threshold = 0.8
    
    async def should_stop_early(self, current_result: str) -> bool:
        # Stop if we have a confident answer
        confidence = await self.calculate_confidence(current_result)
        if confidence >= self.confidence_threshold:
            return True
        
        # Stop if result is comprehensive
        if len(current_result) > 500 and "summary" in current_result.lower():
            return True
            
        return False
    
    async def parallel_tool_execution(self, actions: List[Tuple[str, str]]):
        # Execute multiple tools in parallel when possible
        tasks = []
        for action, input_data in actions:
            if self.is_safe_for_parallel(action):
                task = asyncio.create_task(self.execute_tool(action, input_data))
                tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
```

#### 3. Context-Aware Responses
**Goal**: Make responses more personalized and contextual.

```python
class ContextAwareAgent:
    def __init__(self):
        self.user_profiles = {}
        self.conversation_patterns = {}
    
    async def personalize_response(self, response: str, session_id: str) -> str:
        profile = await self.get_user_profile(session_id)
        
        # Adjust formality based on conversation history
        if profile.get('prefers_formal', False):
            response = self.make_more_formal(response)
        else:
            response = self.make_more_casual(response)
        
        # Add relevant context
        if profile.get('domain_interest'):
            response = self.add_domain_context(response, profile['domain_interest'])
        
        return response
    
    def make_more_formal(self, text: str) -> str:
        # Replace casual phrases with formal ones
        replacements = {
            "hey": "hello",
            "gonna": "going to", 
            "can't": "cannot",
            "it's": "it is"
        }
        
        for casual, formal in replacements.items():
            text = text.replace(casual, formal)
        
        return text
```

---

## LLM Provider System

### The Challenge of Multiple AI Models

Imagine you're building a house and you have access to different contractors:
- **OpenAI (GPT)**: Excellent quality, but expensive and sometimes unavailable
- **Google Gemini**: Fast and cost-effective, great for simple tasks  
- **Hugging Face**: Open source, customizable, but needs more setup

Our **Provider Manager** is like a **smart project manager** that knows which contractor to use for each job.

### Provider Architecture

#### 1. Base Provider Interface
```python
from abc import ABC, abstractmethod

class BaseLLMProvider(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__.replace('Provider', '').lower()
        self.is_available = True
        self.last_error = None
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate chat completion response"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available"""
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate request cost"""
        pass
```

#### 2. OpenAI Provider Implementation
```python
class OpenAIProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.client = AsyncOpenAI(api_key=config['api_key'])
        self.model = config.get('model', 'gpt-4o-mini')
        
        # Pricing (per 1K tokens)
        self.pricing = {
            'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
            'gpt-4': {'input': 0.03, 'output': 0.06}
        }
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                **kwargs
            )
            
            return {
                'response': response.choices[0].message.content,
                'model': self.model,
                'usage': {
                    'input_tokens': response.usage.prompt_tokens,
                    'output_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'cost': self.calculate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            }
            
        except Exception as e:
            self.last_error = str(e)
            self.is_available = "quota" not in str(e).lower()  # Temporary unavailable if quota exceeded
            raise
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            self.is_available = True
            return True
        except Exception as e:
            self.is_available = False
            self.last_error = str(e)
            return False
```

#### 3. Gemini Provider Implementation
```python
class GeminiProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        genai.configure(api_key=config['api_key'])
        self.model = genai.GenerativeModel(config.get('model', 'gemini-1.5-flash'))
        
        # Pricing
        self.pricing = {
            'gemini-1.5-flash': {'input': 0.000075, 'output': 0.0003},
            'gemini-1.5-pro': {'input': 0.00125, 'output': 0.005}
        }
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        try:
            # Convert OpenAI format to Gemini format
            gemini_messages = self.convert_messages(messages)
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                gemini_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1000)
                )
            )
            
            return {
                'response': response.text,
                'model': self.model.model_name,
                'usage': {
                    'input_tokens': response.usage_metadata.prompt_token_count,
                    'output_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                },
                'cost': self.calculate_cost(
                    response.usage_metadata.prompt_token_count,
                    response.usage_metadata.candidates_token_count
                )
            }
            
        except Exception as e:
            self.last_error = str(e)
            raise
    
    def convert_messages(self, messages: List[Dict[str, str]]) -> str:
        # Convert OpenAI chat format to Gemini prompt format
        converted = []
        for msg in messages:
            if msg['role'] == 'system':
                converted.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                converted.append(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                converted.append(f"Assistant: {msg['content']}")
        
        return "\n".join(converted)
```

### Provider Manager: The Orchestrator

#### 1. Core Manager Logic
```python
class LLMProviderManager:
    def __init__(self):
        self.providers = {}
        self.fallback_order = ['gemini', 'openai', 'huggingface']  # Gemini first (free tier)
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = {}
        
        # Load providers from config
        self._load_providers()
    
    def _load_providers(self):
        """Load and initialize all providers"""
        if settings.OPENAI_API_KEY:
            self.providers['openai'] = OpenAIProvider({
                'api_key': settings.OPENAI_API_KEY,
                'model': settings.OPENAI_MODEL
            })
        
        if settings.GEMINI_API_KEY:
            self.providers['gemini'] = GeminiProvider({
                'api_key': settings.GEMINI_API_KEY,
                'model': settings.GEMINI_MODEL
            })
        
        if settings.HUGGINGFACE_API_KEY:
            self.providers['huggingface'] = HuggingFaceProvider({
                'api_key': settings.HUGGINGFACE_API_KEY,
                'model': settings.HUGGINGFACE_MODEL
            })
    
    async def get_best_provider(self, task_type: str = 'general') -> BaseLLMProvider:
        """Select the best available provider for a task"""
        
        # Task-specific preferences
        preferences = {
            'reasoning': ['openai', 'gemini', 'huggingface'],      # Complex reasoning
            'creative': ['openai', 'gemini', 'huggingface'],       # Creative writing
            'factual': ['gemini', 'openai', 'huggingface'],        # Factual queries
            'general': ['gemini', 'openai', 'huggingface']         # General chat
        }
        
        preferred_order = preferences.get(task_type, self.fallback_order)
        
        # Check each provider in preference order
        for provider_name in preferred_order:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                
                # Check if provider is healthy
                if await self._is_provider_healthy(provider):
                    return provider
        
        # If no provider is available, raise exception
        raise RuntimeError("No LLM providers are currently available")
    
    async def _is_provider_healthy(self, provider: BaseLLMProvider) -> bool:
        """Check if provider is healthy (with caching)"""
        current_time = time.time()
        last_check = self.last_health_check.get(provider.name, 0)
        
        # Use cached result if recent
        if current_time - last_check < self.health_check_interval:
            return provider.is_available
        
        # Perform actual health check
        is_healthy = await provider.health_check()
        self.last_health_check[provider.name] = current_time
        
        return is_healthy
```

#### 2. Smart Fallback System
```python
async def generate_response_with_fallback(
    self,
    messages: List[Dict[str, str]],
    provider_name: Optional[str] = None,
    task_type: str = 'general',
    **kwargs
) -> Dict[str, Any]:
    """Generate response with automatic fallback"""
    
    attempted_providers = []
    
    try:
        # Try specific provider if requested
        if provider_name and provider_name in self.providers:
            provider = self.providers[provider_name]
            attempted_providers.append(provider_name)
            
            try:
                return await provider.chat_completion(messages, **kwargs)
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
        
        # Try providers in fallback order
        for provider_name in self.fallback_order:
            if provider_name in attempted_providers:
                continue
                
            if provider_name not in self.providers:
                continue
            
            provider = self.providers[provider_name]
            attempted_providers.append(provider_name)
            
            try:
                logger.info(f"Trying provider: {provider_name}")
                return await provider.chat_completion(messages, **kwargs)
                
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        # All providers failed
        raise RuntimeError(
            f"All LLM providers failed. Attempted: {attempted_providers}"
        )
        
    except Exception as e:
        logger.error(f"LLM generation failed completely: {e}")
        
        # Return a generic fallback response
        return {
            'response': "I apologize, but I'm having trouble processing your request right now. Please try again later.",
            'model': 'fallback',
            'error': str(e),
            'attempted_providers': attempted_providers
        }
```

#### 3. Cost and Usage Tracking
```python
class UsageTracker:
    def __init__(self):
        self.daily_usage = {}
        self.cost_tracking = {}
    
    async def track_request(
        self, 
        provider_name: str, 
        usage: Dict[str, int], 
        cost: float
    ):
        """Track usage and costs"""
        today = datetime.utcnow().date().isoformat()
        
        # Initialize tracking for today if needed
        if today not in self.daily_usage:
            self.daily_usage[today] = {}
            self.cost_tracking[today] = {}
        
        if provider_name not in self.daily_usage[today]:
            self.daily_usage[today][provider_name] = {
                'requests': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'total_tokens': 0
            }
            self.cost_tracking[today][provider_name] = 0.0
        
        # Update usage
        self.daily_usage[today][provider_name]['requests'] += 1
        self.daily_usage[today][provider_name]['input_tokens'] += usage['input_tokens']
        self.daily_usage[today][provider_name]['output_tokens'] += usage['output_tokens']
        self.daily_usage[today][provider_name]['total_tokens'] += usage['total_tokens']
        
        # Update cost
        self.cost_tracking[today][provider_name] += cost
        
        # Log if cost is getting high
        if self.cost_tracking[today][provider_name] > 10.0:  # $10 threshold
            logger.warning(f"High daily cost for {provider_name}: ${self.cost_tracking[today][provider_name]:.2f}")
    
    async def get_usage_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate usage report"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days)
        
        report = {
            'period': f"{start_date} to {end_date}",
            'providers': {},
            'totals': {
                'requests': 0,
                'tokens': 0,
                'cost': 0.0
            }
        }
        
        for date_str, daily_data in self.daily_usage.items():
            date = datetime.fromisoformat(date_str).date()
            if start_date <= date <= end_date:
                for provider, usage in daily_data.items():
                    if provider not in report['providers']:
                        report['providers'][provider] = {
                            'requests': 0,
                            'tokens': 0,
                            'cost': 0.0
                        }
                    
                    report['providers'][provider]['requests'] += usage['requests']
                    report['providers'][provider]['tokens'] += usage['total_tokens']
                    report['providers'][provider]['cost'] += self.cost_tracking[date_str][provider]
                    
                    report['totals']['requests'] += usage['requests']
                    report['totals']['tokens'] += usage['total_tokens']
                    report['totals']['cost'] += self.cost_tracking[date_str][provider]
        
        return report
```

---

## Frontend Implementation

### The User Experience Philosophy

Our frontend follows the principle of **Progressive Disclosure** - show simple features first, advanced features when needed. Think of it like a **smartphone interface** - easy for beginners, powerful for experts.

### Technology Stack Deep Dive

#### 1. Why React + Next.js?
**React**: Component-based, reusable, large ecosystem
**Next.js**: Server-side rendering, API routes, optimized builds, great developer experience

#### 2. TypeScript Benefits
```typescript
// Type safety prevents bugs
interface ChatMessage {
    id: string;
    content: string;
    role: 'user' | 'assistant' | 'system';
    timestamp: Date;
    sentiment?: 'positive' | 'negative' | 'neutral';
    model?: string;
}

// Compile-time error checking
function processMessage(message: ChatMessage) {
    // TypeScript ensures message has all required properties
    console.log(`Processing ${message.role} message: ${message.content}`);
}
```

#### 3. State Management Strategy
We use a **hybrid approach**:
- **React useState** for component-level state
- **Context API** for app-wide state
- **React Query** for server state management

```typescript
// App-wide chat context
interface ChatContextType {
    messages: ChatMessage[];
    currentSession: string;
    isLoading: boolean;
    sendMessage: (content: string) -> Promise<void>;
    uploadFile: (file: File) -> Promise<void>;
    clearSession: () -> void;
}

const ChatContext = createContext<ChatContextType | undefined>(undefined);

export const ChatProvider: React.FC<{children: ReactNode}> = ({children}) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [currentSession, setCurrentSession] = useState<string>('');
    const [isLoading, setIsLoading] = useState(false);
    
    const sendMessage = async (content: string) => {
        setIsLoading(true);
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    message: content,
                    session_id: currentSession
                })
            });
            
            const data = await response.json();
            setMessages(prev => [...prev, data]);
        } catch (error) {
            console.error('Send message failed:', error);
        } finally {
            setIsLoading(false);
        }
    };
    
    return (
        <ChatContext.Provider value={{
            messages,
            currentSession,
            isLoading,
            sendMessage,
            uploadFile,
            clearSession
        }}>
            {children}
        </ChatContext.Provider>
    );
};
```

### Component Architecture

#### 1. Atomic Design Principles
```
Pages (Templates)
├── Chat Page
├── Settings Page
└── Analytics Page

Organisms (Complex Components)  
├── ChatInterface
├── SettingsPanel
└── AnalyticsDashboard

Molecules (Simple Components)
├── MessageBubble  
├── ChatInput
├── FileUpload
└── ProviderSelector

Atoms (Basic Elements)
├── Button
├── Input
├── Avatar
└── LoadingSpinner
```

#### 2. Key Components Implementation

**Chat Interface (Organism)**:
```typescript
const ChatInterface: React.FC = () => {
    const {messages, sendMessage, isLoading} = useChat();
    const messagesEndRef = useRef<HTMLDivElement>(null);
    
    // Auto-scroll to bottom when new messages arrive
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: 'smooth'});
    }, [messages]);
    
    return (
        <div className="flex flex-col h-full">
            {/* Header */}
            <ChatHeader />
            
            {/* Messages Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((message) => (
                    <MessageBubble
                        key={message.id}
                        message={message}
                        isUser={message.role === 'user'}
                    />
                ))}
                {isLoading && <TypingIndicator />}
                <div ref={messagesEndRef} />
            </div>
            
            {/* Input Area */}
            <ChatInput
                onSendMessage={sendMessage}
                disabled={isLoading}
            />
        </div>
    );
};
```

**Message Bubble (Molecule)**:
```typescript
interface MessageBubbleProps {
    message: ChatMessage;
    isUser: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({message, isUser}) => {
    const [showDetails, setShowDetails] = useState(false);
    
    return (
        <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`
                max-w-xs lg:max-w-md px-4 py-2 rounded-lg relative
                ${isUser 
                    ? 'bg-blue-500 text-white' 
                    : 'bg-gray-200 text-gray-800'
                }
            `}>
                {/* Message Content */}
                <div className="whitespace-pre-wrap">
                    {message.content}
                </div>
                
                {/* Timestamp */}
                <div className={`
                    text-xs mt-1 opacity-70
                    ${isUser ? 'text-blue-100' : 'text-gray-500'}
                `}>
                    {formatTimestamp(message.timestamp)}
                </div>
                
                {/* Metadata (expandable) */}
                {!isUser && message.model && (
                    <button
                        onClick={() => setShowDetails(!showDetails)}
                        className="text-xs opacity-50 hover:opacity-100"
                    >
                        {showDetails ? 'Hide' : 'Show'} details
                    </button>
                )}
                
                {showDetails && (
                    <div className="mt-2 pt-2 border-t border-gray-300 text-xs">
                        <div>Model: {message.model}</div>
                        {message.sentiment && (
                            <div>Sentiment: {message.sentiment}</div>
                        )}
                        <div>Processing time: {message.processing_time}s</div>
                    </div>
                )}
            </div>
        </div>
    );
};
```

**Chat Input (Molecule)**:
```typescript
interface ChatInputProps {
    onSendMessage: (message: string) => Promise<void>;
    disabled: boolean;
}

const ChatInput: React.FC<ChatInputProps> = ({onSendMessage, disabled}) => {
    const [input, setInput] = useState('');
    const [isComposing, setIsComposing] = useState(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    
    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!input.trim() || disabled) return;
        
        await onSendMessage(input.trim());
        setInput('');
        
        // Reset textarea height
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };
    
    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
            e.preventDefault();
            handleSubmit(e);
        }
    };
    
    // Auto-resize textarea
    const handleInputChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
        const value = e.target.value;
        setInput(value);
        
        // Auto-resize
        const textarea = e.target;
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    };
    
    return (
        <form onSubmit={handleSubmit} className="p-4 border-t bg-white">
            <div className="flex items-end space-x-2">
                {/* File Upload Button */}
                <FileUploadButton />
                
                {/* Text Input */}
                <div className="flex-1">
                    <textarea
                        ref={textareaRef}
                        value={input}
                        onChange={handleInputChange}
                        onKeyDown={handleKeyDown}
                        onCompositionStart={() => setIsComposing(true)}
                        onCompositionEnd={() => setIsComposing(false)}
                        placeholder="Type your message..."
                        className="w-full p-2 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
                        rows={1}
                        disabled={disabled}
                    />
                </div>
                
                {/* Send Button */}
                <Button
                    type="submit"
                    disabled={!input.trim() || disabled}
                    variant="primary"
                    size="sm"
                >
                    {disabled ? <LoadingSpinner size="sm" /> : <SendIcon />}
                </Button>
            </div>
        </form>
    );
};
```

#### 3. Advanced UI Features

**Real-time Typing Indicator**:
```typescript
const TypingIndicator: React.FC = () => {
    return (
        <div className="flex justify-start">
            <div className="bg-gray-200 rounded-lg p-3">
                <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}} />
                    <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}} />
                </div>
            </div>
        </div>
    );
};
```

**File Upload with Drag & Drop**:
```typescript
const FileUploadArea: React.FC = () => {
    const [isDragOver, setIsDragOver] = useState(false);
    const {uploadFile} = useChat();
    
    const handleDrop = async (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragOver(false);
        
        const files = Array.from(e.dataTransfer.files);
        for (const file of files) {
            if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
                await uploadFile(file);
            }
        }
    };
    
    const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        setIsDragOver(true);
    };
    
    const handleDragLeave = () => {
        setIsDragOver(false);
    };
    
    return (
        <div
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            className={`
                border-2 border-dashed rounded-lg p-8 text-center transition-colors
                ${isDragOver 
                    ? 'border-blue-500 bg-blue-50' 
                    : 'border-gray-300 bg-gray-50'
                }
            `}
        >
            <CloudUploadIcon className="mx-auto h-12 w-12 text-gray-400" />
            <p className="mt-2 text-sm text-gray-600">
                Drag and drop your PDF files here, or click to select files
            </p>
            <input
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={handleFileSelect}
                className="hidden"
                multiple
            />
        </div>
    );
};
```

### Performance Optimizations

#### 1. Code Splitting
```typescript
// Lazy load heavy components
const AnalyticsDashboard = lazy(() => import('./components/AnalyticsDashboard'));
const SettingsPanel = lazy(() => import('./components/SettingsPanel'));

// Use in routes
<Route 
    path="/analytics" 
    element={
        <Suspense fallback={<LoadingSpinner />}>
            <AnalyticsDashboard />
        </Suspense>
    } 
/>
```

#### 2. Virtual Scrolling for Message Lists
```typescript
import { FixedSizeList as List } from 'react-window';

const VirtualizedMessageList: React.FC<{messages: ChatMessage[]}> = ({messages}) => {
    const MessageItem = ({ index, style }: {index: number, style: CSSProperties}) => (
        <div style={style}>
            <MessageBubble
                message={messages[index]}
                isUser={messages[index].role === 'user'}
            />
        </div>
    );
    
    return (
        <List
            height={600}  // Fixed height
            itemCount={messages.length}
            itemSize={80}  // Estimated item height
            itemData={messages}
        >
            {MessageItem}
        </List>
    );
};
```

#### 3. Memoization for Expensive Components
```typescript
const MessageBubble = React.memo<MessageBubbleProps>(({message, isUser}) => {
    // Component implementation...
}, (prevProps, nextProps) => {
    // Custom comparison function
    return (
        prevProps.message.id === nextProps.message.id &&
        prevProps.message.content === nextProps.message.content &&
        prevProps.isUser === nextProps.isUser
    );
});
```

---

## Security & Performance

### Security First Approach

#### 1. API Security
**Authentication & Authorization**:
```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def get_current_user(token: str = Depends(security)) -> User:
    """Verify JWT token and return current user"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        user = await get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user)
):
    # Only authenticated users can access chat
    pass
```

**Rate Limiting**:
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/chat")
@limiter.limit("30/minute")  # 30 requests per minute per IP
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    pass

@app.post("/documents/upload")
@limiter.limit("10/hour")  # 10 uploads per hour per IP
async def upload_document(request: Request, file: UploadFile):
    pass
```

**Input Validation & Sanitization**:
```python
from pydantic import BaseModel, validator
import bleach

class ChatRequest(BaseModel):
    message: str
    session_id: str
    provider: Optional[str] = None
    
    @validator('message')
    def sanitize_message(cls, v):
        # Remove HTML tags and dangerous content
        cleaned = bleach.clean(v, tags=[], strip=True)
        
        # Limit message length
        if len(cleaned) > 5000:
            raise ValueError("Message too long (max 5000 characters)")
        
        return cleaned
    
    @validator('session_id')
    def validate_session_id(cls, v):
        # Ensure session ID is a valid UUID
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            raise ValueError("Invalid session ID format")
```

#### 2. File Upload Security
```python
import magic
import os
from pathlib import Path

ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.md'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

async def validate_uploaded_file(file: UploadFile) -> bool:
    """Comprehensive file validation"""
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not allowed. Allowed: {list(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
        )
    
    # Check MIME type (magic number validation)
    file_content = await file.read(1024)  # Read first 1KB
    await file.seek(0)  # Reset
    
    mime_type = magic.from_buffer(file_content, mime=True)
    
    expected_mimes = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.txt': 'text/plain',
        '.md': 'text/plain'
    }
    
    if mime_type not in expected_mimes.values():
        raise HTTPException(
            status_code=400,
            detail="File content doesn't match file extension"
        )
    
    return True

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    await validate_uploaded_file(file)
    
    # Generate secure filename
    secure_filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join("uploads", secure_filename)
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return {"message": "File uploaded successfully", "file_id": secure_filename}
```

#### 3. Database Security
**SQL Injection Prevention**:
```python
# GOOD - Using SQLAlchemy ORM (automatically parameterized)
def get_user_messages(user_id: str, session_id: str):
    return db.query(ChatMessage).filter(
        ChatMessage.user_id == user_id,
        ChatMessage.session_id == session_id
    ).all()

# BAD - Raw SQL (vulnerable to injection)
# def get_user_messages(user_id: str):
#     query = f"SELECT * FROM messages WHERE user_id = '{user_id}'"
#     return db.execute(query).fetchall()
```

**Sensitive Data Handling**:
```python
import hashlib
from cryptography.fernet import Fernet

class SecurityManager:
    def __init__(self):
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
    
    def hash_sensitive_data(self, data: str) -> str:
        """One-way hash for passwords, etc."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Reversible encryption for API keys, etc."""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()

# Usage
security = SecurityManager()

class User(Base):
    id = Column(String, primary_key=True)
    email = Column(String, unique=True)
    password_hash = Column(String)  # Hashed password
    encrypted_api_keys = Column(Text)  # Encrypted API keys
    
    def set_password(self, password: str):
        self.password_hash = security.hash_sensitive_data(password)
    
    def verify_password(self, password: str) -> bool:
        return self.password_hash == security.hash_sensitive_data(password)
```

### Performance Optimization

#### 1. Database Optimization
**Connection Pooling**:
```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Optimized database engine
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,          # Number of connections to maintain
    max_overflow=30,       # Additional connections when needed
    pool_pre_ping=True,    # Validate connections before use
    pool_recycle=3600      # Recycle connections after 1 hour
)
```

**Query Optimization**:
```python
# GOOD - Eager loading to prevent N+1 queries
def get_session_with_messages(session_id: str):
    return db.query(ChatSession)\
        .options(joinedload(ChatSession.messages))\
        .filter(ChatSession.id == session_id)\
        .first()

# BAD - N+1 query problem
# def get_session_with_messages(session_id: str):
#     session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
#     for message in session.messages:  # This triggers individual queries
#         print(message.content)

# Batch operations
def update_multiple_messages(message_updates: List[Dict]):
    db.bulk_update_mappings(ChatMessage, message_updates)
    db.commit()
```

**Database Indexing**:
```sql
-- Critical indexes for performance
CREATE INDEX CONCURRENTLY idx_messages_session_created 
ON chat_messages(session_id, created_at DESC);

CREATE INDEX CONCURRENTLY idx_messages_role_created 
ON chat_messages(role, created_at DESC);

CREATE INDEX CONCURRENTLY idx_documents_type_status 
ON documents(file_type, processing_status);

-- Full-text search indexes
CREATE INDEX CONCURRENTLY idx_documents_content_fts 
ON documents USING GIN(to_tsvector('english', content));

CREATE INDEX CONCURRENTLY idx_messages_content_fts 
ON chat_messages USING GIN(to_tsvector('english', content));
```

#### 2. Caching Strategy
**Multi-Level Caching**:
```python
import redis
from functools import wraps
import pickle
import asyncio

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expiration: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(expiration=600)  # Cache for 10 minutes
async def get_user_profile(user_id: str) -> Dict[str, Any]:
    # Expensive database query
    return await db.query(User).filter(User.id == user_id).first()

@cache_result(expiration=3600)  # Cache for 1 hour
async def get_document_summary(document_id: str) -> str:
    # Expensive LLM call
    document = await get_document(document_id)
    return await generate_summary(document.content)
```

**Cache Invalidation**:
```python
class CacheManager:
    def __init__(self):
        self.redis = redis.Redis()
    
    async def invalidate_user_cache(self, user_id: str):
        """Invalidate all cache entries for a user"""
        pattern = f"*user:{user_id}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    async def invalidate_session_cache(self, session_id: str):
        """Invalidate cache for a specific session"""
        pattern = f"*session:{session_id}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    async def warm_cache(self, user_id: str):
        """Pre-populate cache with frequently accessed data"""
        # Pre-load user profile
        await get_user_profile(user_id)
        
        # Pre-load recent sessions
        sessions = await get_user_sessions(user_id, limit=5)
        for session in sessions:
            await get_session_history(session.id)
```

#### 3. Async Performance
**Concurrent Processing**:
```python
async def process_multiple_documents(file_paths: List[str]) -> List[str]:
    """Process multiple documents concurrently"""
    
    # Create tasks for concurrent processing
    tasks = []
    for file_path in file_paths:
        task = asyncio.create_task(process_single_document(file_path))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Handle results and exceptions
    processed_ids = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Failed to process {file_paths[i]}: {result}")
        else:
            processed_ids.append(result)
    
    return processed_ids

async def batch_generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings in batches for better performance"""
    batch_size = 32  # Process 32 texts at once
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = await embedding_manager.encode_texts(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

**Background Task Processing**:
```python
from celery import Celery

# Celery for background tasks
celery_app = Celery('chatbot', broker='redis://localhost:6379')

@celery_app.task
def process_document_background(file_path: str, document_id: str):
    """Process document in background"""
    try:
        # Heavy processing
        content = extract_text_from_pdf(file_path)
        chunks = create_text_chunks(content)
        embeddings = generate_embeddings(chunks)
        
        # Save to database
        save_processed_document(document_id, content, chunks, embeddings)
        
        # Notify completion (WebSocket, email, etc.)
        notify_processing_complete(document_id)
        
    except Exception as e:
        logger.error(f"Background processing failed for {document_id}: {e}")
        notify_processing_failed(document_id, str(e))

# Usage in API
@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    # Save file immediately
    document_id = await save_uploaded_file(file)
    
    # Process in background
    process_document_background.delay(file.filename, document_id)
    
    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Document uploaded and processing started"
    }
```

---

## Deployment & Production

### Production Architecture

#### 1. Infrastructure Overview
```
Internet
    ↓
Load Balancer (Nginx)
    ↓
┌─────────────────────────────────────┐
│         Application Layer           │
├─────────────────────────────────────┤
│  API Server 1 (FastAPI + Uvicorn)  │
│  API Server 2 (FastAPI + Uvicorn)  │
│  API Server 3 (FastAPI + Uvicorn)  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│            Data Layer               │
├─────────────────────────────────────┤
│  PostgreSQL (Primary)               │
│  PostgreSQL (Read Replica)          │
│  Redis (Cache + Sessions)           │
│  Elasticsearch (Text Search)        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│          Storage Layer              │
├─────────────────────────────────────┤
│  AWS S3 (Documents + Files)         │
│  Local Storage (Temporary)          │
└─────────────────────────────────────┘
```

#### 2. Docker Configuration

**Dockerfile for Backend**:
```dockerfile
# Multi-stage build for optimization
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Set up working directory
WORKDIR /app
COPY --chown=app:app . .

# Switch to non-root user
USER app

# Make sure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

**Docker Compose for Development**:
```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/chatbot_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
    volumes:
      - ./backend:/app
      - uploads_data:/app/uploads
    restart: unless-stopped

  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: chatbot_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis Cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Frontend (Next.js)
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    volumes:
      - ./frontend:/app
      - /app/node_modules
    restart: unless-stopped

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - backend
      - frontend
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  uploads_data:
```

#### 3. Kubernetes Configuration

**Deployment YAML**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatbot-backend
  labels:
    app: chatbot-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: chatbot-backend
  template:
    metadata:
      labels:
        app: chatbot-backend
    spec:
      containers:
      - name: backend
        image: chatbot-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: chatbot-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: chatbot-config
              key: redis-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: chatbot-backend-service
spec:
  selector:
    app: chatbot-backend
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
```

### Environment Configuration

#### 1. Production Environment Variables
```bash
# .env.production
# Application
NODE_ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-super-secret-key-here
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com

# Database
DATABASE_URL=postgresql://user:password@db-host:5432/chatbot_prod
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=30

# Redis
REDIS_URL=redis://redis-host:6379/0
REDIS_PASSWORD=your-redis-password

# API Keys (encrypted)
OPENAI_API_KEY_ENCRYPTED=encrypted-key-here
GEMINI_API_KEY_ENCRYPTED=encrypted-key-here

# Performance
WORKERS=4
MAX_REQUESTS=1000
MAX_REQUESTS_JITTER=50

# Monitoring
SENTRY_DSN=https://your-sentry-dsn
LOG_LEVEL=info
METRICS_ENABLED=true
```

#### 2. Configuration Management
```python
import os
from typing import List, Optional
from pydantic import BaseSettings, validator

class Settings(BaseSettings):
    # Application
    app_name: str = "Advanced AI Chatbot"
    debug: bool = False
    version: str = "1.0.0"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Security
    secret_key: str
    allowed_origins: List[str] = []
    
    # Database
    database_url: str
    database_pool_size: int = 10
    database_max_overflow: int = 20
    
    # Redis
    redis_url: str
    redis_password: Optional[str] = None
    
    # API Keys (will be decrypted at runtime)
    openai_api_key_encrypted: str
    gemini_api_key_encrypted: str
    
    # Performance
    max_requests: int = 1000
    cache_ttl: int = 300
    
    # Monitoring
    sentry_dsn: Optional[str] = None
    log_level: str = "info"
    
    @validator('allowed_origins', pre=True)
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(',')]
        return v
    
    @property
    def openai_api_key(self) -> str:
        """Decrypt OpenAI API key at runtime"""
        return decrypt_api_key(self.openai_api_key_encrypted)
    
    @property
    def gemini_api_key(self) -> str:
        """Decrypt Gemini API key at runtime"""
        return decrypt_api_key(self.gemini_api_key_encrypted)
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### Monitoring & Observability

#### 1. Application Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('chatbot_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('chatbot_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('chatbot_active_sessions', 'Number of active chat sessions')
LLM_REQUESTS = Counter('chatbot_llm_requests_total', 'LLM requests', ['provider', 'model'])
LLM_ERRORS = Counter('chatbot_llm_errors_total', 'LLM errors', ['provider', 'error_type'])

# Middleware for automatic metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    return response

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# Business metrics
class MetricsCollector:
    @staticmethod
    async def record_llm_request(provider: str, model: str, success: bool, error_type: str = None):
        LLM_REQUESTS.labels(provider=provider, model=model).inc()
        if not success:
            LLM_ERRORS.labels(provider=provider, error_type=error_type).inc()
    
    @staticmethod
    async def update_active_sessions():
        with get_db_session() as db:
            count = db.query(ChatSession).filter(ChatSession.is_active == True).count()
            ACTIVE_SESSIONS.set(count)
```

#### 2. Logging Configuration
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        return json.dumps(log_entry)

# Configure logging
def setup_logging():
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for errors
    file_handler = logging.FileHandler('logs/errors.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

# Usage with context
logger = logging.getLogger(__name__)

async def process_chat_message(message: str, session_id: str, user_id: str):
    logger.info(
        "Processing chat message",
        extra={
            'user_id': user_id,
            'session_id': session_id,
            'message_length': len(message)
        }
    )
    
    try:
        # Process message...
        pass
    except Exception as e:
        logger.error(
            "Chat processing failed",
            extra={
                'user_id': user_id,
                'session_id': session_id,
                'error': str(e)
            },
            exc_info=True
        )
```

#### 3. Health Checks
```python
from fastapi import HTTPException

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    checks = {
        'database': await check_database_health(),
        'redis': await check_redis_health(),
        'llm_providers': await check_llm_providers_health(),
        'disk_space': check_disk_space(),
        'memory_usage': check_memory_usage()
    }
    
    # Determine overall health
    healthy = all(checks.values())
    status_code = 200 if healthy else 503
    
    return {
        'status': 'healthy' if healthy else 'unhealthy',
        'checks': checks,
        'timestamp': datetime.utcnow().isoformat(),
        'version': settings.version
    }

async def check_database_health() -> bool:
    try:
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False

async def check_redis_health() -> bool:
    try:
        redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False

async def check_llm_providers_health() -> bool:
    try:
        healthy_providers = 0
        for name, provider in llm_manager.providers.items():
            if await provider.health_check():
                healthy_providers += 1
        
        # At least one provider should be healthy
        return healthy_providers > 0
    except Exception as e:
        logger.error(f"LLM providers health check failed: {e}")
        return False
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. OpenAI API Quota Issues

**Problem**: `Error 429 - You exceeded your current quota`

**Solution**:
```python
# Check your .env file
DEFAULT_LLM=gemini  # Switch to Gemini as primary

# Monitor usage
async def check_openai_usage():
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    async with aiohttp.ClientSession() as session:
        async with session.get("https://api.openai.com/v1/usage", headers=headers) as response:
            usage_data = await response.json()
            print(f"Current usage: {usage_data}")

# Implement smart fallback
class SmartProviderManager:
    async def get_provider_with_budget_check(self):
        # Check daily spending
        if await self.get_daily_openai_cost() > MAX_DAILY_BUDGET:
            return self.providers['gemini']
        return self.providers['openai']
```

#### 2. Database Connection Issues

**Problem**: `sqlalchemy.exc.DisconnectionError: Connection was invalidated`

**Solution**:
```python
# Add connection pool configuration
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,        # Validate connections
    pool_recycle=3600,         # Recycle every hour
    connect_args={
        "connect_timeout": 10,  # Connection timeout
        "application_name": "chatbot_app"
    }
)

# Add retry logic
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def execute_with_retry(query):
    try:
        with get_db_session() as db:
            return db.execute(query)
    except Exception as e:
        logger.warning(f"Database query failed, retrying: {e}")
        raise
```

#### 3. Memory Issues with Large Documents

**Problem**: `MemoryError` when processing large PDFs

**Solution**:
```python
import gc
from typing import Iterator

class StreamingDocumentProcessor:
    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size
    
    def process_large_document_streaming(self, file_path: str) -> Iterator[str]:
        """Process document in streams to avoid memory issues"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                current_chunk = ""
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    current_chunk += page_text + "\n"
                    
                    # Yield chunk when it reaches max size
                    if len(current_chunk) >= self.max_chunk_size:
                        yield current_chunk
                        current_chunk = ""
                        
                        # Force garbage collection
                        gc.collect()
                
                # Yield remaining chunk
                if current_chunk:
                    yield current_chunk
                    
        except Exception as e:
            logger.error(f"Streaming document processing error: {e}")
            raise

# Usage
async def process_large_pdf(file_path: str):
    processor = StreamingDocumentProcessor()
    
    for chunk in processor.process_large_document_streaming(file_path):
        # Process chunk
        embedding = await embedding_manager.encode_text(chunk)
        await save_chunk_to_database(chunk, embedding)
```

#### 4. Slow Vector Search Performance

**Problem**: Vector search takes too long with many documents

**Solution**:
```python
import faiss

class OptimizedVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        
        # Use IVF (Inverted File) index for better performance
        quantizer = faiss.IndexFlatIP(dimension)
        self.index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
        
        # Enable GPU if available
        if faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
    
    def add_vectors_batch(self, embeddings: List[List[float]], batch_size: int = 1000):
        """Add vectors in batches for better performance"""
        embeddings_np = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_np)
        
        # Train index if not trained
        if not self.index.is_trained:
            self.index.train(embeddings_np[:10000])  # Train on first 10k vectors
        
        # Add in batches
        for i in range(0, len(embeddings_np), batch_size):
            batch = embeddings_np[i:i + batch_size]
            self.index.add(batch)
    
    def search_optimized(self, query_embedding: List[float], k: int = 5):
        """Optimized search with filtering"""
        query_np = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query_np)
        
        # Search with more candidates and filter
        search_k = min(k * 3, self.index.ntotal)  # Search 3x more candidates
        scores, indices = self.index.search(query_np, search_k)
        
        # Filter and return top k
        filtered_results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= SIMILARITY_THRESHOLD and len(filtered_results) < k:
                filtered_results.append((self.documents[idx], float(score)))
        
        return filtered_results
```

#### 5. Agent Timeout Issues

**Problem**: Agent stops due to timeout or iteration limits

**Solution**:
```python
class RobustAgentSystem:
    def __init__(self):
        self.max_iterations = 3
        self.timeout_seconds = 45
        self.fallback_responses = {
            'cv_summary': "I can see you have a CV uploaded. Let me provide a basic summary...",
            'document_search': "I found some documents that might be relevant...",
            'general': "I'm having trouble processing that right now. Could you try rephrasing?"
        }
    
    async def process_with_fallback(self, message: str, session_id: str):
        """Process with robust fallback handling"""
        
        # Determine message type for appropriate fallback
        message_type = self.classify_message_type(message)
        
        try:
            # Try agent processing with timeout
            result = await asyncio.wait_for(
                self.run_agent(message, session_id),
                timeout=self.timeout_seconds
            )
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Agent timeout for session {session_id}")
            return await self.handle_timeout_fallback(message, message_type, session_id)
            
        except AgentIterationLimitError:
            logger.warning(f"Agent iteration limit for session {session_id}")
            return await self.handle_iteration_limit_fallback(message, message_type, session_id)
            
        except Exception as e:
            logger.error(f"Agent error for session {session_id}: {e}")
            return await self.handle_general_fallback(message, message_type, session_id)
    
    def classify_message_type(self, message: str) -> str:
        """Classify message to determine appropriate fallback"""
        message_lower = message.lower()
        
        if any(term in message_lower for term in ['cv', 'resume', 'summarize', 'pdf']):
            return 'cv_summary'
        elif any(term in message_lower for term in ['document', 'file', 'search']):
            return 'document_search'
        else:
            return 'general'
    
    async def handle_timeout_fallback(self, message: str, message_type: str, session_id: str):
        """Handle timeout with direct tool usage"""
        if message_type == 'cv_summary':
            # Directly use document search tool
            doc_tool = DocumentSearchTool()
            doc_result = await doc_tool._arun("CV resume PDF")
            
            if "Found" in doc_result:
                # Generate quick summary
                summary = await self.generate_quick_summary(doc_result)
                return {'output': f"{summary}\n\n*Note: Quick processing due to system load*"}
        
        # Generic LLM fallback
        fallback_response = await llm_manager.generate_response_with_fallback(
            messages=[{"role": "user", "content": message}]
        )
        return {'output': f"{fallback_response['response']}\n\n*Note: Direct response due to timeout*"}
```

#### 6. Frontend Connection Issues

**Problem**: Frontend can't connect to backend API

**Solution**:
```typescript
// API client with retry logic
class APIClient {
    private baseURL: string;
    private maxRetries: number = 3;
    
    constructor(baseURL: string) {
        this.baseURL = baseURL;
    }
    
    async request<T>(
        endpoint: string, 
        options: RequestInit = {},
        retryCount: number = 0
    ): Promise<T> {
        try {
            const response = await fetch(`${this.baseURL}${endpoint}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
            
        } catch (error) {
            if (retryCount < this.maxRetries) {
                // Exponential backoff
                const delay = Math.pow(2, retryCount) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));
                
                return this.request<T>(endpoint, options, retryCount + 1);
            }
            
            throw error;
        }
    }
}

// Usage with error handling
const apiClient = new APIClient(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000');

const useChatAPI = () => {
    const sendMessage = async (message: string, sessionId: string) => {
        try {
            return await apiClient.request<ChatResponse>('/chat', {
                method: 'POST',
                body: JSON.stringify({
                    message,
                    session_id: sessionId
                })
            });
        } catch (error) {
            console.error('Failed to send message:', error);
            
            // Show user-friendly error
            toast.error('Failed to send message. Please check your connection and try again.');
            
            throw error;
        }
    };
    
    return { sendMessage };
};
```

### Performance Debugging

#### 1. Slow Query Identification
```sql
-- Enable slow query logging (PostgreSQL)
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries slower than 1s
SELECT pg_reload_conf();

-- Find slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    rows
FROM pg_stat_statements 
WHERE mean_time > 100  -- Queries with mean time > 100ms
ORDER BY mean_time DESC
LIMIT 10;

-- Analyze query performance
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM chat_messages 
WHERE session_id = 'some-session-id' 
ORDER BY created_at DESC;
```

#### 2. Memory Profiling
```python
import psutil
import tracemalloc

class MemoryProfiler:
    def __init__(self):
        tracemalloc.start()
        self.snapshots = []
    
    def take_snapshot(self, label: str):
        """Take memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
        
        # Current memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        print(f"Memory snapshot '{label}': {memory_mb:.1f} MB")
    
    def compare_snapshots(self, start_label: str, end_label: str):
        """Compare two snapshots"""
        start_snapshot = None
        end_snapshot = None
        
        for label, snapshot in self.snapshots:
            if label == start_label:
                start_snapshot = snapshot
            elif label == end_label:
                end_snapshot = snapshot
        
        if start_snapshot and end_snapshot:
            top_stats = end_snapshot.compare_to(start_snapshot, 'lineno')
            
            print(f"\nTop 10 memory differences ({start_label} -> {end_label}):")
            for stat in top_stats[:10]:
                print(stat)

# Usage
profiler = MemoryProfiler()

async def process_with_profiling(file_path: str):
    profiler.take_snapshot("start")
    
    # Process document
    content = await extract_pdf_content(file_path)
    profiler.take_snapshot("after_extraction")
    
    # Generate embeddings
    embeddings = await generate_embeddings(content)
    profiler.take_snapshot("after_embeddings")
    
    # Save to database
    await save_to_database(content, embeddings)
    profiler.take_snapshot("after_save")
    
    # Compare snapshots
    profiler.compare_snapshots("start", "after_embeddings")
```

---

## Conclusion

Congratulations! You've just learned about building a comprehensive, production-ready AI chatbot system. This guide covered every aspect from basic concepts to advanced production deployment.

### Key Takeaways

1. **Architecture Matters**: Our modular, service-oriented architecture makes the system scalable and maintainable.

2. **AI Integration**: By supporting multiple LLM providers with smart fallbacks, we ensure reliability and cost-effectiveness.

3. **RAG Power**: The combination of document processing, vector embeddings, and intelligent retrieval makes our chatbot truly knowledgeable.

4. **Agent Intelligence**: LangChain agents provide the reasoning capability to use the right tools for each task.

5. **Production Ready**: With proper security, monitoring, caching, and error handling, this system is ready for real-world use.

### Next Steps

1. **Experiment**: Try different embedding models, LLM providers, and agent configurations.

2. **Extend**: Add new tools like web search, email integration, or custom business logic.

3. **Scale**: Implement horizontal scaling, load balancing, and distributed processing.

4. **Monitor**: Set up comprehensive monitoring and alerting for production use.

5. **Optimize**: Profile performance bottlenecks and optimize accordingly.

### Resources for Further Learning

- **LangChain Documentation**: https://docs.langchain.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **FAISS Documentation**: https://faiss.ai/
- **PostgreSQL Performance**: https://wiki.postgresql.org/wiki/Performance_Optimization
- **React Best Practices**: https://react.dev/learn

Remember: Building great AI systems is an iterative process. Start simple, measure everything, and improve continuously.

**Happy building!** 🚀

---

*This documentation was created as a comprehensive guide to understanding and implementing an advanced AI chatbot system. The concepts and patterns shown here can be adapted for various AI applications beyond chatbots.*
