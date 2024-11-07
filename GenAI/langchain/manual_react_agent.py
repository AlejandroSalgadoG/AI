import re
from typing import Any

from langchain.agents import tool
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama
from langchain.tools.render import render_text_description


@tool
def get_text_lenght(text: str) -> int:
    """Returns lenght of a text by characters"""
    print(f"invocation of get_text_length with {text}")
    length = len(text.strip("'\n").strip('"'))
    return length
    # return f"the length of characters in {text} is {length}"


def find_tool(tool_name: str, tools: list[Tool]) -> Tool:
    for tool in tools:
        if tool.name in tool_name:
            return tool
    raise ValueError(f"could not find tool from '{tool_name}'")


def parse_tool_input(raw_tool_input: str) -> str:
    if match := re.search(".*=\s*(.+)", raw_tool_input):
        raw_tool_input = match.group(1)
    return raw_tool_input.replace('"',"")


class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> Any:
        print(f"*** Prompt to LLM ***")
        print(prompts[0])
        print("*******")

    def on_llm_end(
        self, response: LLMResult, **kwargs: Any
    ) -> Any:
        print(f"*** LLM response ***")
        print(response.generations[0][0].text)
        print("*******")


if __name__ == '__main__':
    tools = [get_text_lenght]

    tool_description = render_text_description(tools)
    tool_names = ", ".join([tool.name for tool in tools])

    template = """
Answer the following questions as best you can. You have access to the following tools:

{tools}


Use the following format to show the answer:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

only if the answer is unknown use the following format to process the answer:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

Begin!

Question: {input}
Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=tool_description,
        tool_names=tool_names,
    )

    llm = ChatOllama(
        model="llama3.1",
        stop=["\nObservation"],
        callbacks=[AgentCallbackHandler()],
    )

    intermediate_steps = []
    agent = prompt | llm | ReActSingleInputOutputParser()

    def iter_agent() -> AgentAction | AgentFinish:
        return agent.invoke(
            {
                "input": "What is the length of the word DOG in characters?",
                "agent_scratchpad": format_log_to_str(intermediate_steps),
            }
        )

    agent_step = iter_agent()
    while isinstance(agent_step, AgentAction):
        selected_tool = find_tool(agent_step.tool, tools)
        observation = selected_tool.func(parse_tool_input(agent_step.tool_input))
        intermediate_steps.append((agent_step, str(observation)))
        agent_step = iter_agent()

    print("done")
