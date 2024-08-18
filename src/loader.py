"""Create the RAG based database store."""

import logging
from pathlib import Path
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from utils import populate_source_files


def load_file(filepath: str) -> List[Document]:
    """Extract the information from PDF files."""
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    logging.info("Successfully loaded the doc: %s", filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    return splits


def load_db(root_folder: Path) -> Chroma:
    """Load the vector database and return the chroma instance."""
    files = populate_source_files(root_folder)
    documents: List[Document] = []
    for f in files:
        logging.info("Loading %s", f)
        documents.extend(load_file(f))
    vectorstore = Chroma.from_documents(documents=documents, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_db(Path(__file__).parent.parent)

    
