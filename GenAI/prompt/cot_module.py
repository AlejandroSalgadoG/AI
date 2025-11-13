import dspy

from basic_qa_signature import QuestionSignature
from utils import get_lm


class ThoughtSignature(dspy.Signature):
    question = dspy.InputField(desc="The question to answer")
    thought = dspy.InputField(desc="The previous thought")
    answer = dspy.OutputField(desc="should be between 1 and 5 words")


class DoubleCoT(dspy.Module):
    def __init__(self, question: str):
        self.cot1 = dspy.ChainOfThought(QuestionSignature)
        self.cot2 = dspy.ChainOfThought(ThoughtSignature)

    def forward(self, question: str) -> str:
        thought = self.cot1(question=question)
        answer = self.cot2(question=question, thought=thought.answer)
        return dspy.Prediction(thought=thought.answer, answer=answer.answer)


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)

    question = "What is the second largest city of the birth state of the winner of the nobel prize of peace in 2009?"

    cot_module = DoubleCoT(question=question)
    prediction = cot_module(question=question)

    print(prediction.thought)
    print(prediction.answer)
