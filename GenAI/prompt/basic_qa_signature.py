import dspy

from utils import get_lm


class QuestionSignature(dspy.Signature):
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="should be between 1 and 5 words")


def predict(signature: type[dspy.Signature], question: str) -> str:
    predict = dspy.Predict(signature)
    return predict(question=question)


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)
    prediction = predict(QuestionSignature, question="What is the capital of France?")
    print(prediction.answer)
