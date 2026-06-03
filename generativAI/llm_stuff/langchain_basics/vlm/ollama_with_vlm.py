import base64

from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM, ChatOllama

image_path:str = "./Different-Dog-Breeds-Sitting-on-Grass.jpg"
with open(image_path, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")


llm = ChatOllama(model="gemma4:e4b")

message = HumanMessage(content= [
    {"type": "text", "text": "Describe the image"},
    {"type": "image_url", "image_url": f"data:image/jpg;base64,{image_b64}"}
])

response = llm.invoke([message])

print(response)
