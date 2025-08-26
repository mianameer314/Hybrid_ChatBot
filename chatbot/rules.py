# Simple, fast offline intent rules
import re

from streamlit import text


RULES = [
    {"patterns": [r"\bhi\b", r"\bhello\b", r"\bhey\b"], "reply": "Hello! How can I help today?"},
    {"patterns": [r"\bhours\b", r"open", r"closing"], "reply": "We’re open Mon–Fri, 9am–6pm."},
    {"patterns": [r"\brefund\b", r"\breturn\b"], "reply": "We accept returns within 30 days if unused."},
    {"patterns": [r"\battendance\b"], "reply": "Mark attendance 9:00–10:00 AM in the HR portal."},
    {"patterns": [r"\bstandup\b"], "reply": "Daily standup at 10:30 AM on Google Meet."},
    ]


def offline_answer(text: str) -> str | None:
    t = text.lower()
    for rule in RULES:
        for pat in rule["patterns"]:
            if re.search(pat, t):
                return rule["reply"]
    return None 