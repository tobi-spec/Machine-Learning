from langchain_ollama.llms import OllamaLLM

llm = OllamaLLM(model="mistral")

question = "What is LangChain used for?"
answer = llm.invoke(question)

print(question)
print("Answer: ", answer)