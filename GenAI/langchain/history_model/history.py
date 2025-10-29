from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from json import dumps


def append_file(file_path: str, content: str) -> None:
    with open(file_path, "a") as f:
        f.write(content)


class ChatHistory:
    def __init__(self, system_message: str | None = None):
        self.system_message = system_message
        self.messages: list[BaseMessage] = self._get_init_messages()

    def _get_init_messages(self) -> list[BaseMessage]:
        return [SystemMessage(content=self.system_message)] if self.system_message else []

    def add_user_message(self, message: str) -> None:
        self.messages.append(HumanMessage(content=message))

    def add_user_image(self, image_b64: str, mime: str) -> None:
        self.messages.append(
            HumanMessage(
                content=[
                    {
                        "type": "image",
                        "source_type": "base64",
                        "data": image_b64,
                        "mime_type": mime,
                    }
                ]
            )
        )

    def add_ai_message(self, message: str) -> None:
        self.messages.append(AIMessage(content=message))

    def remove_message(self, idx: int) -> None:
        del self.messages[idx]

    def restart(self) -> None:
        self.messages = self._get_init_messages()

    def get_messages(self) -> dict[str, list[BaseMessage]]:
        return {"messages": self.messages}

    def to_json(self, execution_id: str | None = None) -> dict:
        return {
            "execution_id": execution_id,
            "messages": [
                {"type": message.type, "content": message.content}
                for message in self.messages
            ]
        }

    def log(self, execution_id: str | None = None, file_name: str | None = None) -> None:
        file_name = file_name or "model_calls.jsonl"
        data = self.to_json(execution_id)
        append_file(file_name, dumps(data, ensure_ascii=False) + "\n")
