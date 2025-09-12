from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM, ChatOllama

llm = ChatOllama(model="mistral")

store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(llm, get_session_history)

config = {"configurable": {"session_id": "session1"}}

message = HumanMessage(content="Hi! I'm Bob")
print(message)
response = with_message_history.invoke(message, config=config)
print(response)

message = HumanMessage(content="What's my name?")
print(message)
response = with_message_history.invoke([HumanMessage(content="What's my name?")], config=config)
print(response)
