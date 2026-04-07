from io import BytesIO
from uuid import uuid4

from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGPipeline:
    def __init__(self, database) -> None:
        self.database = database
        self.documents: list = []

    def get_retriever(self):
        if not self.documents:
            return self._get_vectores()

        return EnsembleRetriever(
            retrievers=[self._get_vectores(), self._get_bm25()],
            weights=[0.6, 0.4]
        )

    def _get_vectores(self) -> VectorStoreRetriever:
        return self.database.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def _get_bm25(self):
        return BM25Retriever.from_documents(self.documents, k=12)

    def digest(self, uploaded_file) -> None:
        buf = BytesIO(uploaded_file.getvalue())
        source = DocumentStream(name=uploaded_file.name, stream=buf)
        converter = DocumentConverter()
        result = converter.convert(source)
        markdown = result.document.export_to_markdown()

        if not markdown.strip():
            raise ValueError(f"No text content extracted from {uploaded_file.name}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.split_documents([
            Document(page_content=markdown, metadata={"source": uploaded_file.name})
        ])
        print(f"Indexed {len(docs)} chunks from {uploaded_file.name}")

        self.documents = docs
        ids = [str(uuid4()) for _ in range(len(docs))]
        self.database.add_documents(docs, ids=ids)
