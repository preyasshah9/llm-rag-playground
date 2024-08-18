"""Common Utils."""

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader
import logging
from pathlib import Path
import os


def populate_source_files(parent_folder: Path) -> list[str]:
    logging.info("Loading files from: %s", parent_folder)
    data_source = parent_folder / "data"
    return [os.fspath(f) for f in data_source.glob("*.pdf")]


if __name__ == "__main__":
    files = populate_source_files(Path(__file__).parent.parent)
