import inspect
import json
import os

from langchain.tools import tool
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from pathlib import Path
from typing import Any, Dict, List, Tuple
from uuid_utils import uuid7

YOU_COLOR = "\u001b[94m"
ASSISTANT_COLOR = "\u001b[93m"
RESET_COLOR = "\u001b[0m"


def resolve_absolute_path(path_str: str) -> Path:
    """
    file.py -> /Users/you/project/file.py
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path

@tool
def read_file_tool(filename: str) -> Dict[str, Any]:
    """
    Gets the full content of a file provided by the user.
    :param filename: The name of the file to read.
    :return: The full content of the file.
    """
    full_path = resolve_absolute_path(filename)
    print(full_path)
    with open(full_path, "r") as file:
        content = file.read()
    return {
        "file_path": str(full_path),
        "content": content
    }

@tool
def list_files_tool(path: str) -> Dict[str, Any]:
    """
    Lists the files in a directory provided by the user.
    :param path: The path to a directory to list files from.
    :return: A list of files in the directory.
    """
    full_path = resolve_absolute_path(path)
    all_files = []
    for item in full_path.iterdir():
        all_files.append({
            "filename": item.name,
            "type": "file" if item.is_file() else "directory",
        })
    return {
        "path": str(full_path),
        "files": all_files,
    }

@tool
def edit_file_tool(path: str, old_str: str, new_str: str) -> Dict[str, Any]:
    """
    Replaces first occurrence of old_str with new_str in file. If old_str is empty,
    create/overwrite file with new_str.
    :param path: The path to the file to edit.
    :param old_str: The string to replace.
    :param new_str: The string to replace with.
    :return: A dictionary with the path to the file and the action taken.
    """
    full_path = resolve_absolute_path(path)
    if old_str == "":
        full_path.write_text(new_str, encoding="utf-8")
        return {
            "path": str(full_path),
            "action": "created",
        }
    original = full_path.read_text(encoding="utf-8")
    if original.find(old_str) == -1:
        return {
            "path": str(full_path),
            "action": "old str not found"
        }
    edited = original.replace(old_str, new_str, 1)
    full_path.write_text(edited, encoding="utf-8")
    return {
        "path": str(full_path),
        "action": "edited",
    }

model: Runnable = ChatOllama(model="gemma4:e4b")

agent = create_agent(
    model=model,
    tools=[read_file_tool, list_files_tool, edit_file_tool],
    system_prompt="You are a coding assistant whose goal it is to help us solve coding tasks."
)

while True:
    try:
        user_input = input(f"{YOU_COLOR}You:{RESET_COLOR}:")
    except (KeyboardInterrupt, EOFError):
        break
    config = {"configurable": {"thread_id": str(uuid7())}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    print(f"{ASSISTANT_COLOR}Assistant: {result["messages"][1].content}")
    print(f"{ASSISTANT_COLOR}Tool Calls: {result["messages"][1].tool_calls}")
    print(f"{ASSISTANT_COLOR}Meta Data: {result["messages"][1].usage_metadata}")
