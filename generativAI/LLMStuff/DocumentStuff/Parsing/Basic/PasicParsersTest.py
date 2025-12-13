import unittest
from pathlib import Path

from generativAI.LLMStuff.DocumentStuff.Parsing.Basic.BasicParsers import BasicPDFParser, BasicWordParser, ExcelParser


class MyTestCase(unittest.TestCase):
    def test_pdf_parser(self) -> None:
        parser: BasicPDFParser = BasicPDFParser()
        documents = parser.read("./invoicesample.pdf")
        print(documents)

    def test_word_parser(self) -> None:
        parser: BasicWordParser = BasicWordParser()
        documents = parser.read("./sample.docx")
        print(documents)

    def test_excel_parser(self) -> None:
        parser: ExcelParser = ExcelParser()
        documents = parser.read("./sample.xlsx")
        print(documents)


if __name__ == '__main__':
    unittest.main()


