
"""
lc_rag.py
-------------
Modern Retrieval-Augmented Generation (RAG) script for policy Q&A using LangChain, ChromaDB, and OpenAI.

This script loads company policies, embeds them into a vector store, and answers questions using only the provided context.

Usage:
    python lc_rag.py

References:
    - https://python.langchain.com/docs/
    - https://docs.trychroma.com/
    - https://platform.openai.com/docs/
    - https://docs.llamaindex.ai/
"""

from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter


load_dotenv()

# Embeddings + LLM
emb = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# Load vector store

# Ensure the db directory is absolute and robust
script_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(script_dir, "db")

# Check if the vector store is empty, and if so, embed and add the policy
db = Chroma(
    collection_name="policy",
    embedding_function=emb,
    persist_directory=db_dir,
)

# Check if the collection is empty (no documents)
if not db.get().get('ids'):
    # Read and split the policy file
    policy_path = os.path.join(script_dir, "data", "policies.txt")
    if os.path.exists(policy_path):
        with open(policy_path, encoding="utf-8") as f:
            policy_text = f.read()
        # Split into paragraphs for better retrieval granularity
        paragraphs = [p.strip() for p in policy_text.split("\n\n") if p.strip()]
        docs = [Document(page_content=p) for p in paragraphs]
        db.add_documents(docs)

retriever = db.as_retriever(search_kwargs={"k": 3})

def _format_docs(docs: list[Document]) -> str:
    """
    Concatenate the content of a list of Document objects, separated by double newlines.

    Args:
        docs (list[Document]): List of Document objects.

    Returns:
        str: Combined document content as a single string.
    """
    return "\n\n".join(doc.page_content for doc in docs).strip()

# Modern prompt format
prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
""")

# Build Runnable RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever | RunnableLambda(_format_docs),
        "question": itemgetter("question"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

def answer(question: str) -> str:
    """
    Answer a question using the RAG pipeline, returning a fallback if the answer is not found in context.

    Args:
        question (str): The user's question.

    Returns:
        str: The answer or a fallback message if not found in the knowledge base.
    """
    result = rag_chain.invoke({"question": question})
    fallback_phrases = [
        "The context provided does not include",
        "I cannot answer the question based on the given context",
        "I don't know",
        "I'm sorry",
        "No relevant context"
    ]
    if not result.strip() or any(phrase.lower() in result.lower() for phrase in fallback_phrases):
        return "Sorry, I couldn't find information about that in the current knowledge base."
    return result

if __name__ == "__main__":
    print(answer("What is the refund policy?"))

# to run: python level1/module8_frameworks/lc_rag.py