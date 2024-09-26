from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

if __name__ == "__main__":
    summary_template = """
    given the information {information} about a person I want you to create:
        1. short summary
        2. two interesting facts about them
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOllama(model="llama3.1")

    chain = prompt_template | llm

    information = ""
    res = chain.invoke(input={"information": information})
    print(res)

