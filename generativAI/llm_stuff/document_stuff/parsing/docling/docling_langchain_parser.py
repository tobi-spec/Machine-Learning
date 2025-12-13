from pathlib import Path

from langchain_core.documents import Document
from langchain_docling import DoclingLoader


def resolve_relative_path(path: str) -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent / "sample_documents"
    return base_dir / path

class DoclingLangchainParser:
    def read(self, path: str) -> list[Document]:
        loader = DoclingLoader(resolve_relative_path(path))
        return loader.load()