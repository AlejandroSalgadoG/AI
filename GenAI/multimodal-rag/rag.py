import base64
import re

from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from storage import Storage


prompt_llm = """
You are an analyst tasking with responding user questions.
You will be given some context, use that information to provide a relevant answer

User-provided question: {question}

Context:
{context}
"""


prompt_multimodal = """
You are an analyst tasking with responding user questions.
You will be given a mixed of text and image(s).
Use this information to provide a relevant answer.

User-provided question: {question}

Context:
{context}
"""


class RagLlm:
   def __init__(self):
        self.chain = (
            {
               "context": Storage().get_vector_store_retriver() | RunnableLambda(self.parse_retrived),
               "question": RunnablePassthrough(),
            }
            | PromptTemplate.from_template(prompt_llm)
            | ChatOllama(temperature=0, model="llama3.1")
            | StrOutputParser()
        )

   def parse_retrived(self, data: list[Document]) -> str:
        return "\n".join([d.page_content for d in data])


class RagMultimodal:
    def __init__(self):
        self.last_context = []
        self.chain = (
            {
                "context": Storage().retriever
                | RunnableLambda(self._save_context)
                | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.create_prompt)
            | ChatOllama(temperature=0, model="llama3.2-vision")
            | StrOutputParser()
        )

    def _save_context(self, docs: list[bytes]) -> list[bytes]:
        self.last_context = [doc.decode("utf-8") for doc in docs]
        return docs

    def looks_like_base64(self, data: str) -> bool:
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", data) is not None

    def is_image_data(self, data: str) -> bool:
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(data)[:8]  # Decode and get the first 8 bytes
            for sig, _ in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def split_image_text_types(self, docs: list[bytes]) -> dict[str, list[str]]:
        texts, images_b64 = [], []
        for raw_doc in docs:
            doc = raw_doc.decode("utf-8")
            if self.looks_like_base64(doc) and self.is_image_data(doc):
                images_b64.append(doc)
            else:
                texts.append(doc)
        return {"images": images_b64, "texts": texts}

    def create_prompt(self, data_dict: dict[str, dict[str, list[str]] | str]) -> list[HumanMessage]:
        messages = []

        for image in data_dict["context"]["images"]:
            messages.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

        text_message = {
            "type": "text",
            "text": prompt_multimodal.format(
                context="\n".join(data_dict["context"]["texts"]),
                question=data_dict["question"],
            )
        }
        messages.append(text_message)

        return [HumanMessage(content=messages)]


if __name__ == '__main__':
    from argparse import ArgumentParser

    arg_parser = ArgumentParser(
        prog="rag.py",
        description="script to execute multimodal rag example",
    )
    arg_parser.add_argument("query", help="query to execute")

    args = arg_parser.parse_args()

    rag = RagMultimodal()
    result = rag.chain.invoke(args.query)
    print(result)
