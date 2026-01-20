import os
from io import BytesIO
from operator import itemgetter
from uuid import uuid4

import streamlit as st
from docling.document_converter import DocumentConverter
from docling_core.types.io import DocumentStream
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, Runnable, RunnableConfig, RunnablePassthrough, \
    RunnableLambda, AddableDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# TODOS:
# web scraper
# ragas tests

if "vectordb" not in st.session_state:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    st.session_state["vectordb"] = Chroma(collection_name="example_collection", embedding_function=embeddings, host="localhost")

if "web_context" not in st.session_state:
    st.session_state["web_context"] = "Test my web context."

retriever = st.session_state["vectordb"].as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.35})

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def debug(x: AddableDict) -> AddableDict:
    print("-----------")
    print("Debug Information")
    print("-----------")
    print("Keys:", list(x.keys()))
    print("-----------")
    print("Input:", x["input"])
    print("-----------")
    print("History:")
    for i in x["history"]:
        print(i.content)
    print("-----------")
    print("Context:", x["context"])
    print("-----------")
    return x


web_ctx = st.session_state["web_context"]
def build_context(x: AddableDict) -> str:
    retrieved = format_docs(retriever.invoke(x["input"]))
    if retrieved and web_ctx:
        result = "\n[Retrieved]\n" + retrieved + "\n[Web page]\n" + web_ctx
        print(1)
    elif retrieved:
        result = "\n[Retrieved]\n" + retrieved
        print(2)
    elif web_ctx:
        result = "\n[Web page]\n" + web_ctx
        print(3)
    else:
        result = "None"
        print(4)
    return result


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLChatMessageHistory(f"{session_id}", "sqlite:///chat_history.db")

prompt: Runnable = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("Make short answers, use the following context only when relevant:\n\n{context}"),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

model: Runnable = ChatOllama(model="mistral")

retrieval_chain = ({
    "context": build_context | retriever | format_docs,
    "input": lambda x: x["input"],
    "history": lambda x: x["history"]
}
    | RunnableLambda(debug)
    | prompt
    | model)


chain_with_history = RunnableWithMessageHistory(
    runnable=retrieval_chain,
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

with st.sidebar:
    uploaded_file = st.file_uploader(label="Add to RAG", type=["pdf", "docx", "csv"])

    if uploaded_file is not None:
        buf = BytesIO(uploaded_file.getvalue())
        source = DocumentStream(name=uploaded_file.name, stream=buf)
        converter = DocumentConverter()
        result = converter.convert(source)
        markdown = result.document.export_to_markdown()

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = splitter.split_documents([
            Document(page_content=markdown, metadata={"source": uploaded_file.name})
        ])

        ids = [str(uuid4()) for _ in range(len(docs))]
        st.session_state["vectordb"].add_documents(docs, ids=ids)
        st.success(f"Added {len(docs)} chunks from {uploaded_file.name}")

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
            st.session_state["web_context"] = web_doc[0].page_content
            st.success(f"Added Link to Context")


