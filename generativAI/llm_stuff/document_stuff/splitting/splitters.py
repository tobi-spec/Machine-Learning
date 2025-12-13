from typing import List

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter


class FixedSizeSlidingWindowSplitter:
    def split(self, text: str) -> List[str]:
        splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10, separator=" ") # seperator is important!
        return splitter.split_text(text)


class RecursiveSplitter:
    def split(self, text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        return splitter.split_text(text)

class ParagraphSplitter:
    def split(self, text: str) -> List[str]:
        return text.split("\n")