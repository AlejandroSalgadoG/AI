import uuid

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from definitions import RagDocument, RagDocumentList, WebData


image_summary_prompt = """
You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.
"""


class ImageSummarizer:
    def __init__(self):
        self.model = ChatOllama(model="bakllava")

    def summarize(self, data_b64: str) -> str:
        result = self.model.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": image_summary_prompt},
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
            document = RagDocument(
                uuid=str(uuid.uuid4()),
                summary=self.summarize(web_image.data_b64),
                page_content=web_image.data_b64,
                metadta=web_image.metadata,
            )
            documents.append(document)
        return RagDocumentList(documents)


if __name__ == '__main__':
    from scraper import MultimodalWebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = MultimodalWebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = ImageSummarizer().apply(web_data)
    print(result)
