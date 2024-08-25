"""Utilities to retrieve the data using by calling the locally stored vector database."""

import logging
from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough

from loader import format_docs, load_db
from template import PROMPT_TEMPLATE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vector_database = load_db(Path(__file__).parent.parent)
    retriever = vector_database.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.8, 'k': 3}
    )
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    model = Ollama(model="llama3.1")
    chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
    # User query 
    user_input = input("Enter your question: ")
    response = chain.invoke(user_input)

    # Get the Answer
    logging.info("Answer: %s", response)