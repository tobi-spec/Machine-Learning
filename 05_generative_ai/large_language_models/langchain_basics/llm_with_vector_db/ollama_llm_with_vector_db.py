from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM

documents = [
    Document(page_content="LangChain is a framework for developing applications powered by language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="HuggingFace hosts open-source models and datasets for machine learning.")
]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store_retriever = FAISS.from_documents(documents, embedding_model).as_retriever()

llm = OllamaLLM(model="mistral")

prompt = ChatPromptTemplate.from_template(
    """
            Answer the question using only the context below.
    
            Context:
            {context}
    
            Question:
            {question}
    """
)

rag_chain = ({
                 "context": vector_store_retriever,
                 "question": RunnablePassthrough(),
             }
             | prompt
             | llm)

question = "What is FAISS?"
response = rag_chain.invoke(question)
print(question)
print(response)
