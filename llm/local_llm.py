from llm.base import BaseLLM
from ollama import chat
from ollama import ChatResponse


class LocalLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, temperature: float, top_p: float) -> str:
        response: ChatResponse = chat(model=self.model_name,
                                      messages=[
                                          {
                                              'role': 'user',
                                              'content': prompt,
                                          },
                                      ],
                                      options={
                                          "temperature": temperature,
                                          "top_p": top_p,
                                      })
        return response.message.content
