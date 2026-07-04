from langchain_core.documents import Document

text = "This is a document"

doc = Document(page_content=text,
               metadata= {
                   "source": "test",
                   "mime_type": "text/plain",
               })

print(doc)