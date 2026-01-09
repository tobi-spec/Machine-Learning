import unittest

from generativAI.llm_stuff.rag_stuff.parsing.docling.docling_parsers import *


class MyTestCase(unittest.TestCase):
    def test_pdf_parser(self):
        parser: DoclingParser = DoclingParser()
        document: str = parser.read("./invoicesample.pdf")
        print(document)

    def test_word_parser(self):
        parser: DoclingParser = DoclingParser()
        document: str = parser.read("./sample.docx")
        print(document)

    def test_excel_parser(self):
        parser: DoclingParser = DoclingParser()
        document: str = parser.read("./sample.xlsx")
        print(document)

if __name__ == '__main__':
    unittest.main()
