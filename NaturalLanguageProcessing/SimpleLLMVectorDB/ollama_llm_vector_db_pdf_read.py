from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama.llms import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./pdf/layout-parser-paper.pdf")
documents = []
for page in loader.load():
    documents.append(page)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents, embedding_model)

llm = OllamaLLM(model="mistral")

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = '''What are the authors of the paper LayoutParser: A Unied Toolkit for Deep
Learning Based Document Image Analysis?'''

result = qa_chain.run(query)

print("\n🔍 Question:", query)
print("🧠 Answer:", result)