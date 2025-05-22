from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_ollama import OllamaEmbeddings

from config import configs
from logs import logger


class Storage:
    def __init__(self):
        logger.info("Start initialization of vector storage")
        self.config = configs["storage"]
        self.vector_store = Chroma(
            collection_name="rag",
            embedding_function=OllamaEmbeddings(
                model=self.config.model,
                **self.config.model_params,
            ),
            persist_directory="data/chroma_db",
        )

    def get_vector_store_retriver(self) -> VectorStoreRetriever:
        return self.vector_store.as_retriever(**self.config.extra_params)

    def add_documents(self, docs: list[Document]) -> None:
        logger.info(f"Adding {len(docs)} documents to vector storage")
        self.vector_store.add_documents(docs)

    def _to_label_documents(self, docs: list[Document]) -> list[Document]:
        logger.info("Converting documents to label documents")
        return [
            Document(
                page_content=doc.metadata["label"],
                metadata=doc.metadata,
            )
            for doc in docs
        ]

    def store(self, docs: list[Document]) -> None:
        logger.info("Storing documents")
        self.add_documents(self._to_label_documents(docs))
