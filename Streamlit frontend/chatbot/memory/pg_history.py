from __future__ import annotations
from typing import List
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory.chat_memory import BaseChatMessageHistory
from chatbot.db import get_engine


class PostgresMessageHistory(BaseChatMessageHistory):
    """
    Minimal, robust message history that stores LangChain messages in Postgres.
    Table schema is in the guide (table: chat_messages).
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.engine = get_engine()
        # Ensure table exists (idempotent)
        try:
            with self.engine.begin() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        id BIGSERIAL PRIMARY KEY,
                        session_id VARCHAR(200) NOT NULL,
                        role VARCHAR(20) NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """))
                conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);"
                ))
        except ProgrammingError:
            pass

    # ---- required API ----
    @property
    def messages(self) -> List[BaseMessage]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                text("""
                    SELECT role, content FROM chat_messages
                    WHERE session_id = :sid
                    ORDER BY id ASC
                """),
                {"sid": self.session_id}
            ).fetchall()

        msgs: List[BaseMessage] = []
        for role, content in rows:
            if role == "human":
                msgs.append(HumanMessage(content=content))
            elif role == "ai":
                msgs.append(AIMessage(content=content))
            elif role == "system":
                msgs.append(SystemMessage(content=content))
            else:
                # default fallback
                msgs.append(HumanMessage(content=content))
        return msgs

    def add_message(self, message: BaseMessage) -> None:
        if isinstance(message, HumanMessage):
            role = "human"
        elif isinstance(message, AIMessage):
            role = "ai"
        elif isinstance(message, SystemMessage):
            role = "system"
        else:
            role = "human"

        with self.engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO chat_messages (session_id, role, content)
                    VALUES (:sid, :role, :content)
                """),
                {"sid": self.session_id, "role": role, "content": message.content}
            )

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                text("DELETE FROM chat_messages WHERE session_id = :sid"),
                {"sid": self.session_id}
            )
