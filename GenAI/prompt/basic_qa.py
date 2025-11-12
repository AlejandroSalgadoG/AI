import dspy

from utils import get_lm


def predict(question: str) -> str:
    # Creates the following System message:
    #
    # Your input fields are:
    # 1. `question` (str):
    # Your output fields are:
    # 1. `answer` (str):
    # All interactions will be structured in the following way, with the appropriate values filled in.
    #
    # [[ ## question ## ]]
    # {question}
    #
    # [[ ## answer ## ]]
    # {answer}
    #
    # [[ ## completed ## ]]
    # In adhering to this structure, your objective is:
    #         Given the fields `question`, produce the fields `answer`.

    predict = dspy.Predict("question -> answer")

    # Inputs the following information:
    #
    # [[ ## question ## ]]
    # What is the capital of France?
    #
    # Respond with the corresponding output fields, starting with the field `[[ ## answer ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.

    prediction = predict(question=question)

    # Returns:

    # [[ ## answer ## ]]
    # Paris
    #
    # [[ ## completed ## ]]

    return prediction


if __name__ == "__main__":
    lm = get_lm()
    dspy.settings.configure(lm=lm)
    prediction = predict(question="What is the capital of France?")
    print(prediction.answer)
