from typing import List
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings


#"sentence-transformers/all-mpnet-base-v2"

class HuggingFaceEmbedder:
    def __init__(self, model_name: str) -> None:
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed(self, text: List[str]) -> List[List[float]]:
        return self.embedding_model.embed_documents(text)


class HuggingFaceAllMpnetBaseV2(HuggingFaceEmbedder):
    def __init__(self):
        super().__init__("sentence-transformers/all-mpnet-base-v2")


class HuggingFaceAllMiniLML6V2(HuggingFaceEmbedder):
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")


class HuggingFaceE5BaseV2(HuggingFaceEmbedder):
    def __init__(self):
        super().__init__("intfloat/e5-base-v2")


class FastEmbed:
    def __init__(self) -> None:
        self.embedding_model = FastEmbedEmbeddings()

    def embed(self, text):
        return self.embedding_model.embed_documents(text)


class OllamaNomicEmbed:
    def __init__(self) -> None:
        self.embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    def embed(self, text):
        return self.embedding_model.embed_documents(text)


