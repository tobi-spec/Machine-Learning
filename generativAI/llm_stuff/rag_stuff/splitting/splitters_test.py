import unittest

from generativAI.llm_stuff.document_stuff.splitting.splitters import *

text: str = "LangChain simplifies AI workflows.\n It enables advanced retrieval-augmented generation systems for NLP tasks.\n LangChain simplifies AI workflows.\n"

class MyTestCase(unittest.TestCase):
    def test_FixedSizeSlidingWindowSplitter(self):
        splitter = FixedSizeSlidingWindowSplitter()
        chunks = splitter.split(text=text)
        print(chunks)

    def test_RecursiveSplitter(self):
        splitter = RecursiveSplitter()
        chunks = splitter.split(text=text)
        print(chunks)

    def test_ParagraphSplitter(self):
        splitter = ParagraphSplitter()
        chunks = splitter.split(text=text)
        print(chunks)



if __name__ == '__main__':
    unittest.main()
