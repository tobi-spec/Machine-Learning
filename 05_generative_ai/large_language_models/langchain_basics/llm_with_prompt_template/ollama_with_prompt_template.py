from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from langchain_ollama import ChatOllama


humane_message = HumanMessage(content="I am a human message")
print(humane_message)

ai_message = AIMessage(content="I am a ai message")
print(ai_message)

system_message = SystemMessage(content="I am a system message")
print(system_message)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("You are a helpful assistant."),
    HumanMessagePromptTemplate.from_template("Tell my a joke about {input}")]
)

model = ChatOllama(model="mistral")

chain = prompt.pipe(model) # underlies prompt | model

answer = chain.invoke({"input": "chickens"})
print(answer.content)