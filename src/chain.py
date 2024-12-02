"""Utilities to retrieve the data using by calling the locally stored vector database."""

import logging
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from loader import format_docs, load_db
from template import PROMPT_TEMPLATE


def retrieval_chain(vector_db: Chroma, query_str: str) -> str:
    """Retrieve the Query string."""
    retrieved_docs = vector_db.similarity_search(query_str)
    return format_docs(retrieved_docs)


# Define the LLM chain for generating answers
def llm_chain() -> Any:
    """Define an llm chain using the LLM model and Prompt Template."""
    prompt=PromptTemplate(input_variables=["question", "context"], template=PROMPT_TEMPLATE)
    llm = ChatOpenAI()
    return prompt | llm


def rag_chain(question: str) -> str:
    """RAG Chain."""
    # Retrieve context
    context = retrieval_chain(question)
    # Generate answer using the context
    response = llm_chain.invoke({"context":context, "question":question})
    return response


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vector_database = load_db(Path(__file__).parent.parent)
    user_input = input("Enter your question: ")
    response = rag_chain(user_input)
    # Get the Answer
    logging.info("Answer: %s", response)