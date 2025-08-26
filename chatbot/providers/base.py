from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, user_text: str, system: str = "", stream: bool = False):
        """Yield chunks if stream=True, else return full string."""