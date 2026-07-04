from typing import List

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class FixedSizeSlidingWindowSplitter:
    def __init__(self):
        self.splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=" ") # seperator is important!

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)


class RecursiveSplitter:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    def split(self, text: str) -> List[str]:
        return self.splitter.split_text(text)

class ParagraphSplitter:
    def split(self, text: str) -> List[str]:
        return text.split("\n")