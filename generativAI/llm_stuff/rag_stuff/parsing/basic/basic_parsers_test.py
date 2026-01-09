import unittest

from langchain_core.documents import Document

from generativAI.llm_stuff.rag_stuff.parsing.basic.basic_parsers import *


class MyTestCase(unittest.TestCase):
    def test_pdf_parser(self) -> None:
        parser: BasicPDFParser = BasicPDFParser()
        documents: list[Document] = parser.read("./invoicesample.pdf")
        print(documents)

    def test_word_parser(self) -> None:
        parser: BasicWordParser = BasicWordParser()
        documents: list[Document] = parser.read("./sample.docx")
        print(documents)

    def test_excel_parser(self) -> None:
        parser: ExcelParser = ExcelParser()
        documents: list[Document] = parser.read("./sample.xlsx")
        print(documents)


if __name__ == '__main__':
    unittest.main()


