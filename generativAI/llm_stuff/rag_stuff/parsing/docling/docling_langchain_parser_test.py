import unittest
from typing import List

from langchain_core.documents import Document

from generativAI.llm_stuff.rag_stuff.parsing.docling.docling_langchain_parser import *


class MyTestCase(unittest.TestCase):
    def test_pdf_parser(self):
        parser: DoclingLangchainParser = DoclingLangchainParser()
        document: List[Document] = parser.read("./invoicesample.pdf")
        print(document)

    def test_word_parser(self):
        parser: DoclingLangchainParser = DoclingLangchainParser()
        document: List[Document] = parser.read("./sample.docx")
        print(document)

    def test_excel_parser(self):
        parser: DoclingLangchainParser = DoclingLangchainParser()
        document: List[Document] = parser.read("./sample.xlsx")
        print(document)

if __name__ == '__main__':
    unittest.main()