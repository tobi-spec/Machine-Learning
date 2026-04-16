import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from chatbot_chain import ChatbotChain
from chatbot_chain import ChatHistoryDatabase
from rag_pipeline import RAGPipeline

'''
langchain, at its base, creates a dict which is passed through the chain and get altered by the different components.
Following keys are added during the process:
{
    "context": {context},
    "input": {input},
    "history": {history}
}
'''

session_id = "session1"

if "chatbot" not in st.session_state:
    model = ChatOllama(model="mistral")

    embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_database: VectorStore = Chroma(collection_name="example_collection", embedding_function=embeddings, host="localhost")
    history_database: BaseChatMessageHistory = ChatHistoryDatabase().get_sql_lite(session_id, "sqlite:///chat_history.db")
    rag_retriever = RAGPipeline(vector_database, model).get_retriever()

    st.session_state["chatbot"] = ChatbotChain(model, rag_retriever, history_database)


chatbot: ChatbotChain = st.session_state["chatbot"]
chain = chatbot.retriever_chain_link() | chatbot.prompt_chain_link() | chatbot.model_chain_link()
chain_with_history = chatbot.history_chain_wrapper(chain)


config: RunnableConfig = {"configurable": {"session_id": session_id}}

st.title("Chatbot Application")
history = chatbot.get_history_database()
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
        for chunk in chain_with_history.stream({"input": user_input}, config=config):
            full_response += chunk.content
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

with st.sidebar:
    uploaded_file = st.file_uploader(label="Add to RAG", type=["pdf", "docx", "csv"])

    if uploaded_file is not None:
        st.session_state["vectordb"].digest(uploaded_file)
        st.success(f"Digest {uploaded_file.name}")

    with st.form("webload"):
        link = st.text_input("Enter a link")
        submitted = st.form_submit_button("Submit")

    if submitted:
        link = link.strip()
        if not link:
            st.warning("Enter a URL.")
        elif not link.startswith(("http://", "https://")):
            st.error("Please include http:// or https://")
        else:
            web_doc = WebBaseLoader(link).load()
            chatbot.add_webcontext(web_doc[0].page_content.replace("\n", ""))
            st.success(f"Added Link to Context")
