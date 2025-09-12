from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


documents = [
    Document(page_content="Tomorrow is a important appointment"),
]

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Make short answers, use the following context when relevant:\n\n{context}"
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


model = ChatOllama(model="mistral")

retrieval_chain = {
    "context": itemgetter("input") | retriever | format_docs,
    "input": lambda x: x["input"],
    "history": lambda x: x["history"]
} | prompt | model



store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=retrieval_chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config: RunnableConfig = {"configurable": {"session_id": "session1"}}



print("Hello! My name is Charlie")
for chunk in chain_with_history.stream({"input": "Hello! My name is Charlie"}, config=config):
    if chunk:
        print(chunk.content, end="", flush=True)

print("\n")

print("What's my name?")
for chunk in chain_with_history.stream({"input": "What's my name?"}, config=config):
    if chunk:
        print(chunk.content, end="", flush=True)

print("\n")

print("What is tomorrow?")
for chunk in chain_with_history.stream({"input": "Is tomorrow something important?"}, config=config):
    if chunk:
        print(chunk.content, end="", flush=True)

