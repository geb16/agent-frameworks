# DIRECT RAG (No Frameworks)
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# load DB
db = chromadb.PersistentClient(path="./db")
collection = db.get_or_create_collection("docs")

def embed(text):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def rag(query):
    q_vec = embed(query)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3
    )

    context = "\n\n".join(results["documents"][0])

    prompt = f"""
Answer using ONLY this context.

CONTEXT:
{context}

QUESTION:
{query}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )

    return resp.choices[0].message.content


if __name__ == "__main__":
    print(rag("What is the refund policy?"))
