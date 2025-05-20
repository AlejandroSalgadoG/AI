import uuid

from transformers import pipeline

from config import configs
from definitions import RagDocument, RagDocumentList, WebData
from failback import retry
from logs import logger


class ImageSummarizer:
    def __init__(self):
        logger.info("Initializing image summarizer")
        self.config = configs["image"]
        self.model = pipeline(
            task="image-to-text",
            model=self.config.model,
            **self.config.model_params,
        )

    def summarize(self, urls: list[str]) -> list[str]:
        return [summary[0]["generated_text"] for summary in self.model(urls)]

    @retry(attempts=3)
    def apply(self, web_data: WebData) -> RagDocumentList:
        logger.info("Start to construct image summary")

        summaries = self.summarize(web_data.get_images_urls())
        logger.info("Image summaries completed")

        return RagDocumentList(
            [
                RagDocument(
                    uuid=str(uuid.uuid4()),
                    summary=summary,
                    page_content=web_image.data_b64,
                    metadata=web_image.metadata,
                )
                for web_image, summary in zip(web_data.images, summaries)
            ]
        )


if __name__ == "__main__":
    from scraper import MultimodalWebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = MultimodalWebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = ImageSummarizer().apply(web_data)
    print(result)
