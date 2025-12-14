from pathlib import Path
from docling.document_converter import DocumentConverter

def resolve_relative_path(path: str) -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent / "sample_documents"
    return base_dir / path

class DoclingParser:
    def read(self, path: str) -> str:
        converter = DocumentConverter()
        result = converter.convert(resolve_relative_path(path))
        return result.document.export_to_markdown()
