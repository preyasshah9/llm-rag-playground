"""Create the RAG based database store."""

import logging
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from utils import populate_source_files


def load_file(filepath: str) -> None:
    """Extract the information from PDF files."""
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    logging.info("Successfully loaded the doc: %s", filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    files = populate_source_files(Path(__file__))
    for f in files:
        logging.info("Loading %s", f)
        load_file(f)
