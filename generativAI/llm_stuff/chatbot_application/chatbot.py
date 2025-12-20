import streamlit as st
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, Runnable, RunnableConfig
from langchain_ollama import ChatOllama

model: Runnable = ChatOllama(model="mistral")

prompt: Runnable = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

chain: Runnable = prompt | model

if "lc_store" not in st.session_state:
    st.session_state["lc_store"] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state["lc_store"]
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = "session1"
config: RunnableConfig = {"configurable": {"session_id": session_id}}


st.title("Chatbot Application")

history = get_session_history(session_id)
for message in history.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_input = st.chat_input("Ask me anything!")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in chain_with_history.stream(input={"input": user_input}, config=config):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

