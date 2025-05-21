from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from config import configs
from storage import Storage


class Rag:
    def __init__(self):
        self.config = configs["llm"]
        self.llm = ChatOllama(
            model=self.config.model,
            **self.config.model_params,
        )
        self.last_context = []
        self.retriever = Storage().get_vector_store_retriver()
        self.chain = (
            {
                "context": RunnableLambda(self.get_context),
                "question": RunnablePassthrough(),
            }
            | ChatPromptTemplate.from_template(self.config.prompt)
            | RunnableLambda(self.check)
            | self.llm
            | StrOutputParser()
        )

    def get_context(self, query: str) -> list[Document]:
        documents = self.retriever.invoke(query)
        self.last_context = [document.page_content for document in documents]
        return self.parse_retrived(documents)

    def parse_retrived(self, data: list[Document]) -> str:
        return "\n\n".join([d.page_content for d in data])

    def consult(self, query: str) -> str:
        return self.chain.invoke(query)

    def check(self, data):
        return data


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
