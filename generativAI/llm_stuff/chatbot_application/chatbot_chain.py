from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder, \
    HumanMessagePromptTemplate
from langchain_core.runnables import Runnable, AddableDict, RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever


class ChatHistoryDatabase():
    @staticmethod
    def get_sql_lite(session_id, path) -> BaseChatMessageHistory:
        return SQLChatMessageHistory(session_id, path)

    @staticmethod
    def get_in_memory() -> BaseChatMessageHistory:
        return InMemoryChatMessageHistory()


class ChatbotChain:
    def __init__(self, model, retriever, history_database: BaseChatMessageHistory):
        self.model: Runnable = model
        self.retriever: EnsembleRetriever | VectorStoreRetriever = retriever
        self.history_database: BaseChatMessageHistory = history_database
        self.web_context = None

    def model_chain_link(self) -> Runnable:
        return self.model

    def prompt_chain_link(self) -> Runnable:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template("Make short answers, use the following context only when relevant:\n\n{context}"),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

    def retriever_chain_link(self):
        return {
                "context": self._build_context,
                "input": lambda x: x["input"],
                "history": lambda x: x["history"]
            }

    def _build_context(self, x: AddableDict) -> str:
        retrieved = self._format_docs(self.retriever.invoke(x["input"]))
        if retrieved and self.web_context:
            result = "\n[Retrieved]\n" + retrieved + "\n[Web page]\n" + self.web_context
        elif retrieved:
            result = "\n[Retrieved]\n" + retrieved
        elif self.web_context:
            result = "\n[Web page]\n" + self.web_context
        else:
            result = "None"
        return result

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def debug_chain_link(self, x: AddableDict) -> AddableDict:
        print("-----------")
        print("Debug Information")
        print("-----------")
        print("Keys:", list(x.keys()))
        print("-----------")
        print("Input:", x["input"])
        print("-----------")
        print("History:")
        for i in x["history"]:
            print(i.content)
        print("-----------")
        print("Context:", x["context"])
        print("-----------")
        return x

    def history_chain_wrapper(self, chain: Runnable) -> Runnable:
        return RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=lambda session_id: self.history_database,
            input_messages_key="input",
            history_messages_key="history"
        )

    def add_webcontext(self, context) -> None:
        self.web_context = context

    def get_history_database(self) -> BaseChatMessageHistory:
        return self.history_database
