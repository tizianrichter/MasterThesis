from llm.base import BaseLLM
from openai import OpenAI


class CloudLLM(BaseLLM):
    def __init__(self, model_name: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float = 0.0, top_p: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            top_p=top_p,
        )

        return response.choices[0].message.content.strip()
