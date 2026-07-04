from deepeval.evaluate import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, ChatOllama

from chatbot_chain import ChatbotChain, ChatHistoryDatabase
from rag_pipeline import RAGPipeline


eval_model: OllamaModel = OllamaModel(
    model="gemma4:e4b",
    base_url="http://localhost:11434",
    temperature=0
)
answer_relevancy_metric: AnswerRelevancyMetric = AnswerRelevancyMetric(model=eval_model, threshold=0.5, include_reason=True,  verbose_mode=True)


documents = [
    Document(page_content="LangChain is a framework for developing applications powered by language models."),
    Document(page_content="FAISS is a library for efficient similarity search and clustering of dense vectors."),
    Document(page_content="HuggingFace hosts open-source models and datasets for machine learning.")
]
model = ChatOllama(model="gemma4:e4b")
embeddings: HuggingFaceEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_database: VectorStore = FAISS.from_documents(documents, embeddings)
history_database = ChatHistoryDatabase().get_in_memory()
rag_pipeline = RAGPipeline(vector_database, model)

test_model: ChatbotChain = ChatbotChain(model, rag_pipeline, history_database)
simple_chain = test_model.model_chain_link()

prompt1: str = 'Return the word "Hi"'
simple_chain.invoke(prompt1)



llm_connection: LLMTestCase = LLMTestCase(
  input=prompt1,
  expected_output="Hi",
  actual_output=simple_chain.invoke(prompt1).content
)
assert_test(llm_connection, [answer_relevancy_metric])

# my name is bob - how is my name?

# documents retrieval