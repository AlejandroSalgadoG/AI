from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore
from langchain_chroma import Chroma
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from definitions import UuidDocument


class Storage:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name="multimod_rag",
            embedding_function=OllamaEmbeddings(model="llama3.1"),
            persist_directory="data/chroma_db",
        )

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=LocalFileStore("data/doc_store"),
            id_key="uuid",
        )

    def get_vector_store_retriver(self) -> VectorStoreRetriever:
        return self.retriever.vectorstore.as_retriever()

    def add_vector_info(self, docs: list[UuidDocument]) -> None:
        self.retriever.vectorstore.add_documents(docs)

    def add_doc_info(self, docs: list[UuidDocument]) -> None:
        self.retriever.docstore.mset(
            [(doc.uuid, bytes(doc.page_content, "utf-8")) for doc in docs]
        )

    def add_vector_and_doc_info(
        self, vector_store_docs: list[UuidDocument], doc_store_docs: list[UuidDocument]
    ) -> None:
        self.add_vector_info(vector_store_docs)
        self.add_doc_info(doc_store_docs)
