# LLM-Interface

from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        pass
