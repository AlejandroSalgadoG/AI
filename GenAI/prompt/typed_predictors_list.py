import dspy

from pydantic import BaseModel, Field

from utils import get_lm


class Answer(BaseModel):
    country: str = Field(description="The country of the answer")
    year: int = Field(description="The year of the answer")


class AnswerListSignature(dspy.Signature):
    """Given the user's question, generate the answer with a JSON readable python list"""
    question: str = dspy.InputField(description="The question to answer")
    answer: list[Answer] = dspy.OutputField(description="The answer to the question")


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)

    question = "Generate a list of country and the year of FIFA world cup winners from 2002 to 2022"
    cot = dspy.ChainOfThought(AnswerListSignature)
    prediction = cot(question=question)

    print(prediction.answer)
