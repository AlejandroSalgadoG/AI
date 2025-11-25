import dspy

from utils import get_lm


class Answer(dspy.Signature):
    context = dspy.InputField(description="The context to answer the question")
    question = dspy.InputField(description="The question to answer")
    answer = dspy.OutputField(description="The answer that should be between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        self.retrive = dspy.Retrieve(k=k)
        self.answer = dspy.ChainOfThought(Answer)

    def forward(self, question: str) -> str:
        context = self.retrive(question).passages
        return self.answer(context=context, question=question)


if __name__ == "__main__":
    lm = get_lm()

    # taken from https://dspy.ai/cheatsheet/#dspyretrieve
    # also available at https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.tar.gz
    colbertv2_wiki = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    dspy.settings.configure(lm=lm, rm=colbertv2_wiki)

    question = "What is the date of birth of the player who provided the assist for the final goal in football world cup finals in 2014?"
    rag = RAG()
    answer = rag(question)

    print(answer)
