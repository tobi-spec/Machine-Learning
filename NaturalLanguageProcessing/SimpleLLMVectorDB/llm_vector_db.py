from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

documents = [
    Document(page_content="LangChain is a framework for developing applications powered by language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="HuggingFace hosts open-source models and datasets for machine learning.")
]
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = FAISS.from_documents(documents, embedding_model)

model_id = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")

llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, temperature=0.7)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What is LangChain used for?"
result = qa_chain.run(query)

print("\n🔍 Question:", query)
print("🧠 Answer:", result)