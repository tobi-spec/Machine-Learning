from uuid import uuid4

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = Chroma(collection_name="example_collection", embedding_function=embeddings, host="localhost")

# add items
document_1 = Document(
    page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)
document_2 = Document(
    page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
    metadata={"source": "news"},
    id=2,
)
documents = [document_1, document_2]
uuids = [str(uuid4()) for _ in range(len(documents))]
vector_store.add_documents(documents=documents, ids=uuids)

# retrieve item
results = vector_store.similarity_search_with_score(
    "What was your breakfast?", k=1)
print(results)

# update item
updated_document_1 = Document(
    page_content="I had chocolate chip pancakes and fried eggs for breakfast this morning.",
    metadata={"source": "tweet"},
    id=1,
)
updated_document_2 = Document(
    page_content="The weather forecast for tomorrow is sunny and warm, with a high of 82 degrees.",
    metadata={"source": "news"},
    id=2,
)

updated_documents = [updated_document_1, updated_document_2]
vector_store.update_documents(documents=updated_documents, ids=uuids)

# retrieve item
results = vector_store.similarity_search_by_vector(
    embedding=embeddings.embed_query("I love good weather!"), k=1
)
print(results)

# use vector store as retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 5})
result = retriever.invoke("Stealing from the bank is a crime", filter={"source": "news"})
print(result)