# C) LlamaIndex RAG (Document Indexing + Query Engine)
# li_rag.py
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Configure via Settings (replaces ServiceContext and SimpleNodeParser)
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.text_splitter = SentenceSplitter(chunk_size=300, chunk_overlap=50)

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=3)

if __name__ == "__main__":
    print(query_engine.query("What is the refund policy?"))


