from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

DEFAULT_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def load_rag_prompt():
    try:
        client = Client()
        try:
            return client.pull_prompt("rlm/rag-prompt", dangerously_pull_public_prompt=True)
        except TypeError:
            return client.pull_prompt("rlm/rag-prompt")
    except Exception:
        return DEFAULT_RAG_PROMPT
