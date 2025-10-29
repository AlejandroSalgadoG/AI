from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

from history import ChatHistory
from decorators import retry_and_log


class Model:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float,
        system_message: str | None = None,
        **kwargs,
    ):
        self.chat_history = ChatHistory(system_message)
        self.prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="messages")])
        self.llm = ChatOllama(base_url=base_url, model=model, temperature=temperature)
        self.chain = self.prompt | self.llm
        print(f"model instantiated: {model} with temperature: {temperature}")

    @retry_and_log(
        attepts=3,
        msg_on_error="Error encountered while getting model response, retrying...",
        msg_on_failure="Number of retries exceeded. Stopping.",
    )
    def invoke(self) -> str:
        response = self.chain.invoke(self.chat_history.get_messages())
        return str(response.content)

    def chat(self, message: str) -> str:
        self.chat_history.add_user_message(message)
        response = self.invoke()
        self.chat_history.add_ai_message(response)
        return response

    def restart_chat(self) -> None:
        self.chat_history.restart()

    def log(self, *args, **kwargs) -> None:
        self.chat_history.log(*args, **kwargs)
