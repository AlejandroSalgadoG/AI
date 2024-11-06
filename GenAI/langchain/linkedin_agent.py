from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama


from web_search_tool import get_profile_url_tavily, handle_tavily_response


def linkedin_lookup_agent(name: str) -> str:
    llm = ChatOllama(model="llama3.1")

    template = "given the full name {name} I want you to get me a link to their linkedin profile page. Your answer should contain only a URL"

    prompt_template = PromptTemplate(input_variables=["name"], template=template)

    tools = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url_tavily,
            description="useful to get linkedin profile page",
        ),
        Tool(
            name="extract URL from the LinkedIn profile page",
            func=handle_tavily_response,
            description="useful to extract URL from linkedin profile page",
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name=name)}
    )

    return result["output"]


if __name__ == '__main__':
    linkedin_url = linkedin_lookup_agent(name="Eden Marco Udemy")
    print(linkedin_url)
