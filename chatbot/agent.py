# chatbot/agent.py
import os
from typing import List
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot.rules import offline_answer         # your existing offline rules
from chatbot.sentiments import analyze           # your sentiment fn
from chatbot.memory.pg_history import PostgresMessageHistory

# --- LLM (Gemini via LangChain) ---
def make_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    # 1.5-flash is cheap/fast; you can swap to pro if you like
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

# --- Tools the agent can call ---
def tool_faq(q: str) -> str:
    """Use your existing offline rules as a 'FAQ tool'."""
    ans = offline_answer(q)
    return ans or "No offline match found."

def tool_sentiment(text: str) -> str:
    s = analyze(text)
    return f"Sentiment: {s['label']} (score={s['score']}), Tone: {s['tone']}"

TOOLS: List[Tool] = [
    Tool(
        name="CompanyFAQ",
        description="Answer company FAQs (attendance, standup, holidays, leave, hours, support, refunds, etc.)",
        func=tool_faq,
    ),
    Tool(
        name="SentimentAndTone",
        description="Analyze sentiment and tone of a message.",
        func=tool_sentiment,
    ),
]

# --- Memory (PostgreSQL long-term) ---
def make_memory(session_id: str):
    history = PostgresMessageHistory(session_id=session_id)
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=history,
    )

# --- Create the agent ---
def build_agent(session_id: str):
    llm = make_llm()
    memory = make_memory(session_id)
    agent = initialize_agent(
        tools=TOOLS,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=False,
        handle_parsing_errors=True,
    )
    return agent
