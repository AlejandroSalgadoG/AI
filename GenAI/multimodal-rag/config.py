import re

from dataclasses import dataclass


@dataclass
class Config:
    model: str
    prompt: str
    model_params: dict

    def __post_init__(self):
        setattr(self, "prompt", re.sub(r"\n +", "\n", self.prompt))


@dataclass
class TextConfig(Config):
    split_params: dict


configs = {
    "storage": Config(
        model="llama3.1",
        model_params={
            "temperature": 0,
        },
        prompt="",
    ),
    "text": TextConfig(
        model="llama3.1",
        model_params={
            "temperature": 0,
        },
        prompt="""
            You are an assistant tasked with summarizing text for retrieval.
            These summaries will be embedded and used to retrieve the raw text elements.
            Give a concise summary of the text that is well optimized for retrieval.
            Text: {element}

            {format_instructions}
        """,
        split_params={
            "chunk_size": 2000,
            "chunk_overlap": 0,
            "separators": ["\n", "."],
            "keep_separator": False,
        },
    ),
    "image": Config(
        model="bakllava",
        model_params={},
        prompt="""
            You are an assistant tasked with summarizing images for retrieval.
            These summaries will be embedded and used to retrieve the raw image.
            Give a concise summary of the image that is well optimized for retrieval.
        """,
    ),
    "llm": Config(
        model="llama3.1",
        model_params={
            "temperature": 0,
        },
        prompt="""
            You are an analyst tasking with responding user questions.
            You will be given some context, use that information to provide a relevant answer

            User-provided question: {question}

            Context:
            {context}
        """,
    ),
    "multimodal": Config(
        model="llama3.2-vision",
        model_params={
            "temperature": 0,
        },
        prompt="""
            You are an analyst tasking with responding user questions.
            You will be given a mixed of text and image(s).
            Use this information to provide a relevant answer.

            User-provided question: {question}

            Context:
            {context}
        """,
    ),
}
