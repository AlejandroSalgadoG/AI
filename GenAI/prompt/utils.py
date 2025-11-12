import dspy


def get_lm():
    return dspy.LM(
        "ollama_chat/llama3.1",
        api_base="http://localhost:11434",
        api_key="",
    )
