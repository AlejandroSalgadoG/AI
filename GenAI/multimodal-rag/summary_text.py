import uuid

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from config import configs
from definitions import RagDocument, RagDocumentList, WebData
from failback import retry
from logs import logger


class Summary(BaseModel):
    content: str = Field(description="summary")

    def to_dict(self) -> dict[str, str]:
        return {"summary": self.content}


class TextSummarizer:
    def __init__(self):
        logger.info("Initializing text summarizer")
        self.config = configs["text"]
        self.model = ChatOllama(model=self.config.model, **self.config.model_params)
        self.summary_parser = PydanticOutputParser(pydantic_object=Summary)
        self.text_splitter = RecursiveCharacterTextSplitter(**self.config.split_params)

    def summarize(self, text_chunks: list[str]) -> list[Summary]:
        prompt = PromptTemplate.from_template(
            self.config.prompt,
            partial_variables={
                "format_instructions": self.summary_parser.get_format_instructions()
            },
        )
        chain = {"element": lambda x: x} | prompt | self.model | self.summary_parser
        return chain.batch(text_chunks, {"max_concurrency": 5})

    @retry(attempts=3)
    def apply(self, web_data: WebData) -> RagDocumentList:
        source = web_data.text.metadata["source"]
        logger.info(f"Start to construct text summaries for {source}")

        text_chunks = self.text_splitter.split_text(web_data.text.get_data_with_title())
        logger.info(f"Text divided in {len(text_chunks)} chunks")

        summaries = self.summarize(text_chunks)
        logger.info("Text summaries completed")

        return RagDocumentList(
            [
                RagDocument(
                    uuid=str(uuid.uuid4()),
                    summary=summary.content,
                    page_content=text,
                    metadata=web_data.text.metadata,
                )
                for text, summary in zip(text_chunks, summaries)
            ]
        )


if __name__ == "__main__":
    from scraper import MultimodalWebLoader

    articles_url = [
        "https://www.deeplearning.ai/the-batch/the-difference-between-ai-safety-and-responsible-ai/"
    ]
    loader = MultimodalWebLoader(web_paths=articles_url)
    [web_data] = loader.load_web_paths()

    result = TextSummarizer().apply(web_data)
    print(result)
