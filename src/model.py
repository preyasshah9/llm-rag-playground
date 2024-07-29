"""Create the RAG based database store."""

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
from pathlib import Path
import os


def extract_data_from_source() -> list[str]:
    data_source = Path(__file__).parent.parent / "data"
    return [os.fspath(f) for f in data_source.glob("*.pdf")]


if __name__ == "__main__":
    files = extract_data_from_source()
