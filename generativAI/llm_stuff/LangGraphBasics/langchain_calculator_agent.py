from langchain.agents import initialize_agent, AgentType
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="mistral")

tools = load_tools(["llm-math"], llm=llm)

agent = initialize_agent(
    tools,
    llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True
)

response = agent.invoke("Jeryy has 3 apples. He buys 2 more apples and then gives 4 apples to his friend. How many apples does he have now?")
print(response)