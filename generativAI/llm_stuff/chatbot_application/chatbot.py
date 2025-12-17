import streamlit as st
from langchain_ollama import ChatOllama

model = ChatOllama(model="mistral")

st.title("Chatbot Application")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask me anything!")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in model.stream(prompt):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

