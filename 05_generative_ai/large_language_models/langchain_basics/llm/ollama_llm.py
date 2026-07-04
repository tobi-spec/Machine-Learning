from langchain_ollama import ChatOllama
from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="mistral")
question = "What is Ollama?"
answer = llm.invoke(question)

print(question)
print("Answer:", answer)

model = ChatOllama(model="mistral")
question = "What is Ollama?"
answer = model.invoke(question)

print(question)
print("Answer:", answer)