from langchain_ollama import ChatOllama

model = ChatOllama(model="mistral")
for chunk in model.stream("Write a poem about the sea."):
    if chunk:
        print(chunk.content, end="", flush=True)