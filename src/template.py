"""A class representing the prompt template."""

from typing import Final


PROMPT_TEMPLATE: Final = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {input} 
Context: {context} 
Answer:
"""
