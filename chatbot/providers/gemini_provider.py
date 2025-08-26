import google.generativeai as genai
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)


    def generate(self, user_text: str, system: str = "", stream: bool = False):
        prompt = f"{system}\n\nUser: {user_text}\nAssistant:"
        if stream:
            for ev in self.model.generate_content(prompt, stream=True):
                if getattr(ev, "text", None):
                    yield ev.text
        else:
            out = self.model.generate_content(prompt)
            return out.text or "(No response)"