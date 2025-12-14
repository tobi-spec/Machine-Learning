import unittest

from generativAI.llm_stuff.document_stuff.embedding.embedding_models import *

texts = ['LangChain simplifies AI workflows.', ' It enables advanced retrieval-augmented generation systems for NLP tasks.', ' LangChain simplifies AI workflows.']

class MyTestCase(unittest.TestCase):
    def test_HuggingFaceAllMpnetBaseV2(self):
        embedder: HuggingFaceAllMpnetBaseV2 = HuggingFaceAllMpnetBaseV2()
        print(embedder.embed(texts))

    def test_HuggingFaceAllMiniLML6V2(self):
        embedder: HuggingFaceAllMiniLML6V2 = HuggingFaceAllMiniLML6V2()
        print(embedder.embed(texts))

    def test_HuggingFaceE5BaseV2(self):
        embedder: HuggingFaceE5BaseV2 = HuggingFaceE5BaseV2()
        print(embedder.embed(texts))

    def test_FastEmbed(self):
        embedder: FastEmbed = FastEmbed()
        print(embedder.embed(texts))

    def test_OllamaNomicEmbed(self):
        embedder: OllamaNomicEmbed = OllamaNomicEmbed()
        print(embedder.embed(texts))

if __name__ == '__main__':
    unittest.main()
