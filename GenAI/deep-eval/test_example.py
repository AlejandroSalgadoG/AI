from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )

    template = """
    tell me an interest fact about the city {city}.
    """

    prompt_template = PromptTemplate(input_variables=["city"], template=template)
    llm = ChatOllama(model="llama3.1")

    chain = prompt_template | llm

    city = "rome"
    res = chain.invoke(input={"city": city})

    test_case = LLMTestCase(
        input=prompt_template.format(city=city),
        actual_output=res.content,
        expected_output=f"an interesting fact about {city} is",
    )

    assert_test(test_case, [correctness_metric])
