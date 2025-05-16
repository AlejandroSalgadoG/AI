from deepeval import assert_test  # type: ignore[import-untyped]
from deepeval.metrics import (  # type: ignore[import-untyped]
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # type: ignore[import-untyped]

from rag import RagMultimodal


def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )

    rag = RagMultimodal()
    query = "what do you know about Voice Stack?"
    result = rag.chain.invoke(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=result,
        expected_output="Voice Stack refers to the collection of technologies and systems used for building voice-based applications",
    )

    assert_test(test_case, [correctness_metric])


def test_rag():
    rag = RagMultimodal()
    query = "what do you know about Voice Stack?"
    actual_output = rag.chain.invoke(query)

    ans_relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        include_reason=True,
    )

    faithfulness_metric = FaithfulnessMetric(
        threshold=0.5,
        include_reason=True,
    )

    relevancy_metric = ContextualRelevancyMetric(
        threshold=0.5,
        include_reason=True,
    )

    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=rag.last_context,
    )

    assert_test(test_case, [ans_relevancy_metric, faithfulness_metric, relevancy_metric])
