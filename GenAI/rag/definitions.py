from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WebData:
    text: str
    metadata: dict[str, str]

    def get_data_with_title(self) -> str:
        if title := self.metadata.get("title"):
            return title + "\n" + self.text
        return self.text
