from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama


def linkedin_lookup_agent(name: str) -> str:
    llm = ChatOllama(model="llama3.1")

    template = "given the full name {name} I want you to get me a link to their linkedin profile page. Your answer should contain only a URL"

    prompt_template = PromptTemplate(input_variables=["name"], template=template)

    tools = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func="?",
            description="useful to get linkedin page URL",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name=name)}
    )

    return result["output"]
