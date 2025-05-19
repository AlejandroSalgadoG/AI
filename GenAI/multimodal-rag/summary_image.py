import uuid

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import configs
from definitions import RagDocument, RagDocumentList, WebData
from logs import logger


class ImageSummarizer:
    def __init__(self):
        logger.info("Initializing image summarizer")
        self.config = configs["image"]
        self.model = ChatOllama(model=self.config.model, **self.config.model_params)

    def summarize(self, data_b64: str) -> str:
        result = self.model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": self.config.prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{data_b64}"},
                        },
                    ]
                )
            ]
        )
        return result.content

    def apply(self, web_data: WebData) -> RagDocumentList:
        documents = []
        for web_image in web_data.images:
            url = web_image.metadata["url"]
            logger.info(f"Start to construct image summary for {url}")
            document = RagDocument(
                uuid=str(uuid.uuid4()),
                summary=self.summarize(web_image.data_b64),
                page_content=web_image.data_b64,
                metadta=web_image.metadata,
            )
            logger.info(f"Summary: {document.summary}")
            documents.append(document)
        return RagDocumentList(documents)


if __name__ == "__main__":
    from scraper import MultimodalWebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = MultimodalWebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = ImageSummarizer().apply(web_data)
    print(result)
