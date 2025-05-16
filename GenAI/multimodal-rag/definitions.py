from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


class UuidDocument(Document):
    uuid: str

    def __init__(self, page_content: str, uuid: str, **kwargs: Any) -> None:
        super().__init__(page_content=page_content, uuid=uuid, **kwargs)


class RagDocument(UuidDocument):
    summary: str

    def __init__(
        self,
        page_content: str,
        uuid: str,
        summary: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            page_content=page_content,
            uuid=uuid,
            summary=summary,
            **kwargs,
        )

    def get_content_doc(self) -> UuidDocument:
        metadata = self.metadata.copy()
        metadata["uuid"] = self.uuid
        return UuidDocument(
            uuid=self.uuid,
            page_content=self.page_content,
            metadata=metadata,
        )

    def get_summary_doc(self) -> UuidDocument:
        metadata = self.metadata.copy()
        metadata["uuid"] = self.uuid
        return UuidDocument(
            uuid=self.uuid,
            page_content=self.summary,
            metadata=metadata,
        )


@dataclass
class RagDocumentList:
    documents: list[RagDocument]

    def get_content_docs(self) -> list[UuidDocument]:
        return [doc.get_content_doc() for doc in self.documents]

    def get_summary_docs(self) -> list[UuidDocument]:
        return [doc.get_summary_doc() for doc in self.documents]


@dataclass
class MultimodalData:
    source: str
    text: Document
    images: list[RagDocument]
    text_parts: list[RagDocument]


@dataclass
class WebImage:
    data_b64: str
    metadata: dict[str, str]


@dataclass
class WebText:
    data: str
    metadata: dict[str, str]

    def get_data_with_title(self) -> str:
        if title := self.metadata.get("title"):
            return title + "\n" + self.data
        return self.data


@dataclass
class WebData:
    text: WebText
    images: list[WebImage]
