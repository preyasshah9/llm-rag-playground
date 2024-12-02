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


def format_docs(docs) -> str:
    for doc in docs:
        print(doc.page_content)
    return "\n\n".join(doc.page_content for doc in docs)

def load_file(filepath: str) -> List[Document]:
    """Extract the information from PDF files."""
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    logging.info("Successfully loaded the doc: %s", filepath)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    splits = text_splitter.split_documents(docs)
    print(f"Splitting Doc into {len(splits)} chunks")
    return splits


def get_embedding_function(model_name: str) -> HuggingFaceEmbeddings:
    """Embedding function to use for embedding the vector tokens."""
    return HuggingFaceEmbeddings(model_name=model_name)


def load_db(root_folder: Path) -> Chroma:
    """Load the vector database and return the chroma instance."""
    files = populate_source_files(root_folder)
    documents: List[Document] = []
    for f in files:
        logging.info("Loading %s", f)
        split_documents = load_file(f)
        documents.extend(split_documents)
    vectorstore = Chroma.from_documents(documents=documents, embedding=get_embedding_function(model_name="all-MiniLM-L6-v2"))
    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_db(Path(__file__).parent.parent)

    
