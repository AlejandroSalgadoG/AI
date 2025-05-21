from deepeval import evaluate, metrics
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase

from rag import Rag
from test_dataset import golden_data

goldens = [
    Golden(input=data["question"], expected_output=data["expected_answer"])
    for data in golden_data
]

dataset = EvaluationDataset(goldens=goldens)
rag = Rag()

test_cases = [
    LLMTestCase(
        input=golden.input,
        actual_output=rag.consult(golden.input),
        expected_output=golden.expected_output,
        retrieval_context=rag.last_context,
    )
    for golden in goldens
]

evaluate(
    test_cases,
    metrics=[
        metrics.AnswerRelevancyMetric(),
        metrics.FaithfulnessMetric(),
        metrics.ContextualPrecisionMetric(),
        metrics.ContextualRecallMetric(),
        metrics.ContextualRelevancyMetric(),
    ],
)
