import dspy

from basic_qa_signature import predict, QuestionSignature
from utils import get_lm


def cot_predict(signature: type[dspy.Signature], question: str) -> str:
    # System message:

    # Your input fields are:
    # 1. `question` (str): The question to answer
    # Your output fields are:
    # 1. `reasoning` (str):
    # 2. `answer` (str): should be between 1 and 5 words
    # All interactions will be structured in the following way, with the appropriate values filled in.
    #
    # [[ ## question ## ]]
    # {question}
    #
    # [[ ## reasoning ## ]]
    # {reasoning}
    #
    # [[ ## answer ## ]]
    # {answer}
    #
    # [[ ## completed ## ]]
    # In adhering to this structure, your objective is:
    #         Given the fields `question`, produce the fields `answer`.

    cot_predict = dspy.ChainOfThought(signature)

    # input:

    # [[ ## question ## ]]
    # Who provided the assist for the final goal in football world cup finals in 2014?
    #
    # Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.

    cot_prediction = cot_predict(question=question)

    # result
    #
    # [[ ## reasoning ## ]]
    # To determine who provided the assist for the final goal in football world cup finals in 2014, we need to refer to historical records of that event. In the 2014 World Cup Final between Germany and Argentina, Mario Götze scored the winning goal in extra time.
    #
    # [[ ## answer ## ]]
    # Mario Gotze's teammate
    #
    # [[ ## completed ## ]]

    return cot_prediction


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)

    question = "What is the birth state of the winner of the nobel prize of peace in 2009?"

    prediction = predict(QuestionSignature, question=question)
    print(prediction.answer)  # should be Hawaii but probably will be wrong

    cot_prediction = cot_predict(QuestionSignature, question=question)
    print(cot_prediction.reasoning)
    print(cot_prediction.answer)
