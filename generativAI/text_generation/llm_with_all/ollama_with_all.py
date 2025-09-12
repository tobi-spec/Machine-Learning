from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableConfig
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

model = ChatOllama(model="mistral")

chain = prompt | model



store = {}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

config: RunnableConfig = {"configurable": {"session_id": "session1"}}

print(chain_with_history.invoke({"input": "Hello! My name is Charlie"}, config=config))
print(chain_with_history.invoke({"input": "What's my name?"}, config=config))


