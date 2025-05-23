import uuid

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import configs
from definitions import WebData
from failback import retry
from logs import logger


class ImageHandler:
    def __init__(self):
        logger.info("Initializing image handler")
        self.config = configs["image"]
        self.model = ChatOllama(model=self.config.model, **self.config.model_params)

    @retry(attempts=3)
    def summarize(self, data_b64: str) -> str:
        return self.model.invoke(
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
        ).content

    def apply(self, web_data: WebData) -> list[Document]:
        documents = []
        for web_image in web_data.images:
            logger.info(f"Start to construct image summary for {web_image.url}")
            document = Document(
                page_content=web_image.data_b64,
                metadata={
                    "label": self.summarize(web_image.data_b64),
                    "url": web_image.url,
                    "uuid": str(uuid.uuid4()),
                },
            )
            logger.info(f"Summary: {document.metadata['label']}")
            documents.append(document)
        return documents


if __name__ == "__main__":
    from scraper import WebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = WebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = ImageHandler().apply(web_data)
    print(result)
