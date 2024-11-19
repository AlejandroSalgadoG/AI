from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

from chains import generate_chain, reflect_chain


REFLECT = "reflect"
GENERATE = "generate"


def generation_node(state: Sequence[BaseMessage]) -> AIMessage:
    res = generate_chain.invoke({"messages": state})
    return res


def reflection_node(state: Sequence[BaseMessage]) -> list[BaseMessage]:
    res = reflect_chain.invoke({"messages": state})
    return [HumanMessage(content=res.content)]


def should_continue(state: Sequence[BaseMessage]) -> str:
    return END if len(state) > 3 else REFLECT


tweet = """
Make this tweet better:
@LangChainAI
- newly Tool Calling feature is seriously underrated.

Agter a long wait, it's here- making the implementation of agents across different models with function calling - super easy

Made a video covering their newest blog post
"""


if __name__ == "__main__":
    builder = MessageGraph()

    builder.add_node(GENERATE, generation_node)
    builder.add_node(REFLECT, reflection_node)
    builder.add_conditional_edges(GENERATE, should_continue)
    builder.add_edge(REFLECT, GENERATE)

    builder.set_entry_point(GENERATE)

    graph = builder.compile()
    # graph.get_graph().draw_mermaid()

    inputs = HumanMessage(content=tweet)
    response = graph.invoke(inputs)
    print(response)