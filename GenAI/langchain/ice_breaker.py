from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from linkedin_scraper import scrape_linkedin_profile
from linkedin_agent import linkedin_lookup_agent

if __name__ == "__main__":
    summary_template = """
    given the Linkedin information {information} about a person I wany you to create:
        1. short summary
        2. two interesting facts about them
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(model="llama3.1")

    chain = prompt_template | llm

    linkedin_url = linkedin_lookup_agent(name="Eden Marco Udemy")
    linkedin_data = scrape_linkedin_profile(linkedin_url)
    res = chain.invoke(input={"information": linkedin_data})
    print(res)

