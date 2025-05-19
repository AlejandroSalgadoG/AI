text_summary_prompt = """
You are an assistant tasked with summarizing text for retrieval.
These summaries will be embedded and used to retrieve the raw text elements.
Give a concise summary of the text that is well optimized for retrieval.
Text: {element}

{format_instructions}
"""


image_summary_prompt = """
You are an assistant tasked with summarizing images for retrieval.
These summaries will be embedded and used to retrieve the raw image.
Give a concise summary of the image that is well optimized for retrieval.
"""


prompt_llm = """
You are an analyst tasking with responding user questions.
You will be given some context, use that information to provide a relevant answer

User-provided question: {question}

Context:
{context}
"""


prompt_multimodal = """
You are an analyst tasking with responding user questions.
You will be given a mixed of text and image(s).
Use this information to provide a relevant answer.

User-provided question: {question}

Context:
{context}
"""
