import uuid

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import configs
from definitions import WebData
from logs import logger


class TextHandler:
    def __init__(self):
        logger.info("Initializing text handler")
        self.config = configs["text"]
        self.text_splitter = RecursiveCharacterTextSplitter(**self.config.extra_params)

    def apply(self, web_data: WebData) -> list[Document]:
        source = web_data.metadata["source"]
        logger.info(f"Starting to process {source}")

        text_chunks = self.text_splitter.split_text(web_data.get_data_with_title())
        logger.info(f"Text divided in {len(text_chunks)} chunks")

        return [
            Document(
                page_content=text,
                metadata=dict(
                    label=text,
                    uuid=str(uuid.uuid4()),
                    **web_data.metadata,
                ),
            )
            for text in text_chunks
        ]


if __name__ == "__main__":
    from scraper import WebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = WebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = TextHandler().apply(web_data)
    print(result)
