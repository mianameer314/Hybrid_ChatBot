# app.py (Streamlit)
import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

# custom chatbot imports
from chatbot.config import settings
from chatbot.rules import offline_answer
from chatbot.providers.gemini_provider import GeminiProvider
from chatbot.providers.hf_provider import HFLocalProvider
from chatbot.agent import build_agent  # LangChain Agent

# --- Load .env secrets ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or getattr(settings, "GEMINI_API_KEY", None)

# --- Backend API URL ---
API_URL = os.getenv("API_URL")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title=getattr(settings, "app_title", "AI Chatbot"),
    page_icon="🤖",
    layout="centered",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp { background:#f8f9fa; }
    .bot {
        background:#e3f2fd;
        color:#000;
        padding:10px 14px;
        border-radius:14px;
        margin:6px 0;
        max-width:72%;
        word-wrap:break-word;
    }
    .user {
        background:#1976d2;
        color:#fff;
        padding:10px 14px;
        border-radius:14px;
        margin:6px 0 6px auto;
        max-width:72%;
        word-wrap:break-word;
    }
    .meta {
        font-size: 12px;
        opacity: 0.85;
        margin-top: 2px;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        margin-right: 6px;
        background: #eef2ff;
        color: #1e40af;
    }
    .chat-container {
        min-height:400px;
        display:flex;
        flex-direction:column;
        justify-content:flex-end;
        padding:10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Init ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I’m a hybrid bot 🤖. I can work offline, with Gemini, HuggingFace, or in Agent mode (LangChain + PostgreSQL)."}
    ]

# --- Sidebar Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    demo_mode = st.toggle("Demo mode (offline only)", value=False)
    model_choice = st.radio(
        "Mode",
        ["Hybrid (Gemini → HF)", "Gemini only", "HuggingFace only", "Agent (LangChain)"],
        index=0
    )
    session_id = st.text_input("Session ID", value=st.session_state.session_id)

    # Clear history in DB + Redis
    if st.button("Clear history"):
        st.session_state.messages = [
            {"role": "assistant", "content": "History cleared. How can I help?"}
        ]
        try:
            requests.post(f"{API_URL}/clear/{session_id}")     # DB + Redis invalidation server-side
        except Exception:
            pass
        st.rerun()

    st.divider()
    st.subheader("🗄 Redis Tools")

    if st.button("🔌 Check Connection"):
        try:
            resp = requests.get(f"{API_URL}/cache/ping", timeout=5)
            if resp.status_code == 200:
                st.success(f"connected: {resp.json()}")
            else:
                st.error(f"❌ Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"⚠️ Could not connect: {str(e)}")

    if st.button("📜 View History"):
        try:
            resp = requests.get(f"{API_URL}/history/{session_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    st.info("Messages (from cache or DB):")
                    for msg in data:
                        line = f"**{msg['role']}**: {msg['content']}"
                        if msg.get("sentiment_label"):
                            line += f"  _(sentiment: {msg['sentiment_label']}, tone: {msg.get('tone')})_"
                        st.write(line)
                else:
                    st.warning("No messages yet.")
            else:
                st.error(f"❌ Error: {resp.status_code} - {resp.text}")
        except Exception as e:
            st.error(f"⚠️ Could not fetch history: {str(e)}")

# --- Backend Helpers ---
def load_history(session_id: str):
    try:
        res = requests.get(f"{API_URL}/history/{session_id}")
        if res.status_code == 200:
            return res.json()
    except Exception:
        return []
    return []

def save_message(session_id: str, role: str, content: str):
    try:
        payload = {"session_id": session_id, "role": role, "content": content}
        requests.post(f"{API_URL}/send", json=payload)
    except Exception:
        pass

# --- Sync with DB ---
backend_history = load_history(session_id)
if backend_history:
    st.session_state.messages = backend_history

# --- Show Chat History ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for m in st.session_state.messages:
    css = "bot" if m["role"] == "assistant" else "user"
    st.markdown(f"<div class='{css}'>{m['content']}</div>", unsafe_allow_html=True)

    # show sentiment meta under USER messages if available
    if m["role"] == "user":
        badges = []
        if m.get("sentiment_label"):
            badges.append(f"<span class='badge'>Sentiment: {m['sentiment_label']}</span>")
        if m.get("tone"):
            badges.append(f"<span class='badge'>Tone: {m['tone']}</span>")
        if m.get("sentiment_score") is not None:
            badges.append(f"<span class='badge'>Score: {round(float(m['sentiment_score']), 3)}</span>")
        if badges:
            st.markdown(f"<div class='meta'>{' '.join(badges)}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input ---
if user := st.chat_input("Type a message…"):
    # Save immediately to backend (backend computes & stores sentiment)
    save_message(session_id, "user", user)

    reply = None

    # Agent mode (LangChain + PostgreSQL memory)
    if model_choice == "Agent (LangChain)":
        try:
            agent = build_agent(session_id=session_id)
            # Provide last messages as context
            history = [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
            history_text = "\n".join(history[-10:])
            reply = agent.run(f"Conversation so far:\n{history_text}\nUser: {user}")
        except Exception as e:
            reply = f"⚠️ Agent error: {e}"

    else:
        # 1) Offline rules
        reply = offline_answer(user)

        # 2) Gemini
        if not reply and not demo_mode and GEMINI_API_KEY and model_choice in ["Hybrid (Gemini → HF)", "Gemini only"]:
            provider = GeminiProvider(GEMINI_API_KEY)
            try:
                with st.chat_message("assistant"):
                    ph = st.empty()
                    buf = ""
                    history = [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
                    history_text = "\n".join(history[-10:])
                    for chunk in provider.generate(
                        f"Conversation so far:\n{history_text}\nUser: {user}",
                        system="Be concise, friendly, and helpful.",
                        stream=True
                    ):
                        buf += chunk
                        ph.markdown(buf)
                    reply = buf
            except Exception as e:
                reply = f"⚠️ Gemini error: {e}"

        # 3) HuggingFace local
        if not reply and not demo_mode and model_choice in ["Hybrid (Gemini → HF)", "HuggingFace only"]:
            try:
                hf_provider = HFLocalProvider(model_name="distilgpt2")
                history = [f"{m['role']}: {m['content']}" for m in st.session_state.messages]
                history_text = "\n".join(history[-10:])
                reply = hf_provider.generate(f"{history_text}\nUser: {user}")
            except Exception as e:
                reply = f"⚠️ HF provider error: {e}"

    # Final fallback
    if not reply:
        reply = "🤔 I don’t know that yet. Try Agent mode or enable Gemini."

    # Add assistant message locally + persist
    st.session_state.messages.append({"role": "assistant", "content": reply})
    try:
        requests.post(f"{API_URL}/send", json={"session_id": session_id, "role": "assistant", "content": reply})
    except Exception:
        pass

    # Refresh history (to pull stored sentiment for the user message we just sent)
    st.rerun()

