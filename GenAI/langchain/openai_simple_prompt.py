from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

if __name__ == "__main__":
    summary_template = """
    given the information {information} about a person from I wany you to create:
        1. short summary
        2. two interesting facts about them
    """

    prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = prompt_template | llm

    information = ""
    res = chain.invoke(input={"information": information})
    print(res)
