import re

from dataclasses import dataclass


@dataclass
class Config:
    model: str
    prompt: str
    model_params: dict
    extra_params: dict

    def __post_init__(self):
        setattr(self, "prompt", re.sub(r"\n +", "\n", self.prompt))


configs = {
    "text": Config(
        model="",
        prompt="",
        model_params={},
        extra_params={
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "add_start_index": True,
        },
    ),
    "storage": Config(
        model="llama3.2",
        model_params={
            "temperature": 0,
        },
        extra_params={},
        prompt="",
    ),
    "llm": Config(
        model="qwen2.5",
        model_params={
            "temperature": 0,
        },
        extra_params={},
        prompt="""
            You are an AI Assisant. Use the following context to answer the question correctly.
            If you dont know the answer, just tell, I dont know.

            "context: {context} \n\n"
            "question: {question} \n\n"
            "AI answer:
        """,
    ),
}
