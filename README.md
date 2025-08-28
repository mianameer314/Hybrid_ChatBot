# 🤖 Advanced Agentic Chatbot

A comprehensive AI chatbot system with **RAG**, **Multi-LLM support**, **Sentiment Analysis**, and **LangChain Agents**.

## ✨ Features

### 🚀 **Core Capabilities**
- **Multi-LLM Support**: OpenAI GPT, Google Gemini, HuggingFace models
- **RAG System**: Document processing (PDF, DOCX, TXT), web scraping, vector embeddings
- **Sentiment Analysis**: Advanced emotion detection using HuggingFace transformers
- **LangChain Agents**: Intelligent agents with tools and reasoning
- **Conversation Memory**: Persistent chat history with database storage
- **Real-time Analytics**: Sentiment tracking, conversation insights

### 🛠 **Technical Stack**
- **Backend**: FastAPI, PostgreSQL, Redis, SQLAlchemy
- **Frontend**: Streamlit with modern UI components
- **AI/ML**: OpenAI, Gemini, HuggingFace, Sentence Transformers, FAISS
- **Document Processing**: PyPDF2, python-docx, BeautifulSoup
- **Vector Search**: FAISS for semantic similarity

## ✨ Features

### 1. 🔄 **Conversational Memory**

* The chatbot **remembers past conversations** within the session.
* Uses `ConversationBufferMemory` (LangChain).
* This lets it give **contextual replies** (not just one-off answers).

**Example:**

```
User: What is FastAPI?  
Bot: FastAPI is a Python framework for building APIs quickly.  
User: And who created it?  
Bot: It was created by Sebastián Ramírez.  
```

---

### 2. 📊 **Sentiment Analysis Integration**

* Every user message passes through `sentiments.py` before reaching the agent.
* Uses **NLTK’s VADER** sentiment analyzer.
* Labels messages as **Positive, Negative, or Neutral**.
* Future use cases: customizing bot tone, analytics dashboard, user feedback trends.

---

### 3. ⚙️ **LangChain Agent with Tools**

* The **agent** is initialized in `chatbot/agent.py`.
* It uses LangChain’s `initialize_agent` to connect with:

  * **LLMs** (OpenAI GPT, Gemini).
  * **Custom tools** (e.g., knowledge base, sentiment analyzer).
* Currently supports **ReAct-style agent execution**.

---

### 4. 🖥️ **Streamlit Frontend**

* User-friendly chat UI built with Streamlit.
* Features:

  * Chat bubbles for user & bot.
  * Sidebar with connection settings.
  * Auto-scroll & rerun for real-time conversation.
* Updated from **`st.experimental_rerun()` → `st.rerun()`** (since Streamlit deprecated the old method).

---

### 5. 🗄️ **Database Persistence (Optional)**

* Backend supports PostgreSQL via `DATABASE_URL`.
* Conversations, users, or analytics can be persisted.
* Local setup uses `postgresql+psycopg2`.

---

### 6. 🌐 **API Backend (FastAPI)**

* Exposes REST endpoints for the chatbot.
* Example:

  * `POST /chat` → send a user message & get reply.
* Deployed on **Railway** (`API_URL` in `.env`).
* Streamlit frontend calls this API to get responses.

---

### 7. 🔑 **Multi-LLM Support**

* Supports both **OpenAI** and **Google Gemini**.
* Keys stored in `.env` →

  ```env
  OPENAI_API_KEY=sk-...
  GEMINI_API_KEY=AIza...
  ```
* Switch between models via config.

---

## 🛠️ Recent Changes

1. ✅ **Sentiment Module**

   * Moved `sentiments.py` into **backend services** (`app/services/sentiments.py`).
   * Streamlit frontend imports from backend → `from app.services.sentiments import analyze`.

2. ✅ **Streamlit Rerun Fix**

   * Old: `st.experimental_rerun()` → ❌ deprecated.
   * New: `st.rerun()` → ✅ works in Streamlit v1.30+.

3. ✅ **LangChain Deprecation Warnings**

   * `agent.run()` → marked for deprecation.
   * Should migrate to `.invoke()` (future-proof, but not breaking yet).

4. ✅ **Torch Warnings**

   * PyTorch warnings (`_register_pytree_node`) are safe to ignore.

---

## 🚀 How It Works (Step by Step)

1. **User enters a message** in Streamlit.
2. Message → Sentiment Analyzer (`analyze` in `sentiments.py`).
3. Sentiment result stored (can be logged, displayed, or affect tone).
4. Message → LangChain Agent (`agent.py`).
5. Agent queries the chosen LLM (OpenAI/Gemini) & tools.
6. Reply returned → displayed in Streamlit chat bubble.
7. Memory ensures next turn keeps context.

---

## 📊 Architecture Diagram (Mermaid)

```mermaid
flowchart TD
    A[👤 User in Streamlit] --> B[📊 Sentiment Analyzer (VADER)]
    B --> C[⚙️ LangChain Agent]
    C --> D[🤖 LLMs (OpenAI / Gemini)]
    D --> C
    C --> E[💬 Bot Reply]
    E --> A
```

---

## 🖥️ Setup & Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-repo/chatbot_project.git
cd chatbot_project
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Setup `.env` File

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
DATABASE_URL=postgresql+psycopg2://postgres:mypass@localhost:5432/chatbot_agent
API_URL=https://chatbotbackend-production-xxxx.up.railway.app/chat
```

### 5. Run Backend (FastAPI)

```bash
uvicorn app.main:app --reload
```

### 6. Run Frontend (Streamlit)

```bash
cd "StreamLit Frontend"
streamlit run app.py
```

---

## 📌 Next Steps (Future Improvements)

* Migrate from **LangChain Agent → LangGraph** for more robust state handling.
* Store **sentiment results in DB** for analytics dashboards.
* Add **user authentication** (different users, personalized memory).
* Improve UI with themes & conversation history export.
