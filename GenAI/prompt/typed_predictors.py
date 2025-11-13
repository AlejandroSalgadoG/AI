import dspy

from pydantic import BaseModel, Field

from utils import get_lm


class AnswerConfidnece(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="The confidence in the answer between 0 and 1")


class QuestionConfidenceSignature(dspy.Signature):
    question: str = dspy.InputField(description="The question to answer")
    answer: AnswerConfidnece = dspy.OutputField(description="The answer and confidence in the answer")


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)

    question = "Who was the winner of the nobel prize of peace in 2009?"

    cot = dspy.ChainOfThought(QuestionConfidenceSignature)
    prediction = cot(question=question)

    print(prediction.answer)
