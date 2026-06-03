from urllib.request import urlopen

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_quickjs import CodeInterpreterMiddleware
from deepagents.backends import StateBackend

checkpointer = MemorySaver()
backend = StateBackend()

skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
with urlopen(skill_url) as response:
    skill_content = response.read().decode('utf-8')

skills_files = {
"/skills/langgraph-docs/SKILL.md": create_file_data(skill_content),
}

llm = ChatOllama(model="gemma4:e4b")

agent = create_deep_agent(
    model=llm,
    backend=backend,
    skills=["/skills/"],
    checkpointer=checkpointer,
    middleware=[CodeInterpreterMiddleware(skills_backend=backend)]
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "What is langgraph?"}],
        # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
        "files": skills_files,
    },
    config={"configurable": {"thread_id": "12345"}},
)

print(result)