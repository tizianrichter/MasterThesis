from llm.base import BaseLLM
from google import genai


class CloudLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        return response.text
