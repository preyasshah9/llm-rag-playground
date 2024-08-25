"""A class representing the prompt template."""

from typing import Final


PROMPT_TEMPLATE: Final = """
You are an assistant for question-answering tasks.
Use the provided context only to answer the following question:

{context}

Question: {input}
"""
