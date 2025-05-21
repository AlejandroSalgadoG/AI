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
