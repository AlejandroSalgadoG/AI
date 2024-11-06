from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from linkedin_scraper import scrape_linkedin_profile

if __name__ == "__main__":
    summary_template = """
    given the Linkedin information {information} about a person I wany you to create:
        1. short summary
        2. two interesting facts about them
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(model="llama3.1")

    chain = prompt_template | llm

    linkedin_data = scrape_linkedin_profile("https://www.linkedin.com/in/eden-marco")
    res = chain.invoke(input={"information": linkedin_data})
    print(res)

