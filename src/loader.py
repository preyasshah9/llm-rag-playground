"""Create the RAG based database store."""

import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from utils import populate_source_files


def load_file(filepath: str) -> None:
    """Extract the information from PDF files."""
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    logging.info("Successfully loaded the doc: %s", filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=GPT4AllEmbeddings())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    files = populate_source_files(Path(__file__))
    for f in files:
        logging.info("Loading %s", f)
        load_file(f)
