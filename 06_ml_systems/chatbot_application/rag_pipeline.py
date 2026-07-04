from io import BytesIO
from typing import Any
from uuid import uuid4
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever, VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ConfigDict


class HydeRetriever(BaseRetriever):
    database: VectorStore
    model: Runnable
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        hyde_prompt: ChatPromptTemplate = ChatPromptTemplate.from_template(
            "Given the following question, generate a hypothetical passage that would answer this question:\nQuestion:{query}\nhypothetical Passage:"
        )
        hyde_chain = hyde_prompt | self.model
        hypothetical_prompt = hyde_chain.invoke({"query": query})
        hypothetical_text = getattr(hypothetical_prompt, "content", str(hypothetical_prompt))
        return self.database.similarity_search(hypothetical_text, k=4)


class RAGPipeline:
    def __init__(self, database: VectorStore, model) -> None:
        self.database: VectorStore = database
        self.model: Runnable = model
        self.documents: list = []

    def get_retriever(self) -> EnsembleRetriever | VectorStoreRetriever:
        if not self.documents:
            return EnsembleRetriever(
                retrievers=[self._get_vectors(), self._get_hyde_vectors()],
                weights=[0.5, 0.5]
        )

        return EnsembleRetriever(
            retrievers=[self._get_vectors(), self._get_bm25(), self._get_hyde_vectors()],
            weights=[0.4, 0.3, 0.3]
        )

    def _get_vectors(self) -> VectorStoreRetriever:
        return self.database.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    def _get_bm25(self) -> BM25Retriever:
        return BM25Retriever.from_documents(self.documents, k=12)

    def _get_hyde_vectors(self) -> HydeRetriever:
        return HydeRetriever(database=self.database, model=self.model)


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


