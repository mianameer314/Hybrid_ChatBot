# chatbot/providers/hf_provider.py
from transformers import pipeline

class HFLocalProvider:
    def __init__(self, model_name: str = "distilgpt2"):  
        """
        Hugging Face Local Provider
        Default model: distilgpt2 (safe GPT-style model).
        """
        self.generator = pipeline(
            "text-generation",
            model="distilgpt2",
            revision="main",  # lock to avoid fallback
            device=-1         # force CPU (avoid CUDA/torch class errors if GPU not setup)
        )

    def generate(self, prompt: str, system: str = "", stream: bool = False):
        text = f"{system}\nUser: {prompt}\nAssistant:"
        result = self.generator(
            text,
            max_length=200,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        reply = result[0]["generated_text"].replace(text, "").strip()
        return reply
