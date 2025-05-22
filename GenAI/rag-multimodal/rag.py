import base64
import re

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from config import configs
from storage import Storage


def looks_like_base64(document: Document) -> bool:
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", document.page_content) is not None


def get_image_document_format(document: Document) -> str | None:
    image_signatures = {
        b"\xff\xd8\xff": "image/jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "image/png",
        # b"\x47\x49\x46\x38": "gif",
        # b"\x52\x49\x46\x46": "webp",
    }
    try:
        # Decode and get the first 8 bytes
        header = base64.b64decode(document.page_content)[:8]
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return format
        return None
    except Exception:
        return None


def split_image_text_documents(documents: list[Document]) -> dict[str, list[Document]]:
    texts, images = [], []
    for document in documents:
        if looks_like_base64(document):
            if format := get_image_document_format(document):
                document.metadata["format"] = format
                images.append(document)
        else:
            texts.append(document)
    return {"images": images, "texts": texts}


class Rag:
    def __init__(self):
        self.config = configs["llm"]
        self.llm = ChatOllama(
            model=self.config.model,
            **self.config.model_params,
        )
        self.last_context = []
        self.storage = Storage()
        self.retriever = self.storage.retriever
        self.chain = (
            {
                "info": RunnableLambda(self.get_info),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.create_llm_input)
            | self.llm
            | StrOutputParser()
        )

    def get_info(self, query: str) -> dict[str, list[Document]]:
        documents = self.retriever.invoke(query)
        return split_image_text_documents(documents)

    def create_context(self, documents: list[Document]) -> str:
        contents = [document.page_content for document in documents]
        self.last_context = contents
        return "\n\n".join(contents)

    def create_llm_input(self, data: dict) -> list[HumanMessage]:
        messages: list[str | dict] = []

        for image in data["info"]["images"]:
            messages.append(
                {
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": image.metadata["format"],
                    "data": image.page_content,
                },
            )

        messages.append(
            {
                "type": "text",
                "text": self.config.prompt.format(
                    context=self.create_context(data["info"]["texts"]),
                    question=data["question"],
                ),
            }
        )

        return [HumanMessage(content=messages)]

    def consult(self, query: str) -> str:
        return self.chain.invoke(query)


if __name__ == "__main__":
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        prog="rag.py",
        description="script to execute multimodal rag example",
    )
    arg_parser.add_argument(
        "query",
        help="query to execute",
        nargs="?",
        type=str,
        default="what do you know about microsoft phi-4-reasoning?",
    )

    args = arg_parser.parse_args()

    result = Rag().consult(args.query)
    print(result)
