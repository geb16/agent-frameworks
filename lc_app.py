# app.py
"""
Streamlit RAG Chatbot (Policy Q&A)
"""

from typing import List, Tuple
from langchain_core.documents import Document
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ---------- 1. Page config & basic styling ---------- #

st.set_page_config(
    page_title="Policy RAG Chatbot",
    page_icon="💬",
    layout="wide",
)

# Light custom CSS to make it feel more “app-like”
st.markdown(
    """
    <style>
    .main {
        background: radial-gradient(circle at top left, #f9fafb, #e5e7eb);
    }
    .chat-bubble-user {
        background-color: #2563eb;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 1rem 1rem 0.25rem 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
    }
    .chat-bubble-assistant {
        background-color: white;
        color: #111827;
        padding: 0.75rem 1rem;
        border-radius: 1rem 1rem 1rem 0.25rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e5e7eb;
        max-width: 80%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- 2. RAG components (cached) ---------- #

@st.cache_resource
def get_rag_components() -> Tuple[Chroma, ChatOpenAI, ChatPromptTemplate, StrOutputParser]:
    """
    Initialise embeddings, Chroma vector store, LLM and prompt.

    This runs once per app session thanks to @st.cache_resource.
    """

    load_dotenv()  # loads OPENAI_API_KEY and others from .env if present

    # Embeddings must match the ones used during ingestion
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Use absolute path for persist_directory based on this script's location

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_dir = os.path.join(script_dir, "db")
    db = Chroma(
        collection_name="policy",
        embedding_function=embeddings,
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

    # Chat model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

    # Prompt with clear instructions
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant that answers questions ONLY using the provided context.
If the answer is not in the context, say you don't know and suggest checking the policy.

Context:
{context}

Question:
{question}

Answer (be clear, concise and user-friendly):
"""
    )

    parser = StrOutputParser()

    return db, llm, prompt, parser


def format_context(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs).strip()


def run_rag(question: str) -> Tuple[str, List[Document]]:
    """
    Execute a single RAG round trip:
    - Retrieve documents
    - Build context
    - Run prompt + LLM
    - Return answer + source docs
    """
    db, llm, prompt, parser = get_rag_components()

    retriever = db.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    if not docs:
        return (
            "I couldn't find anything in the policy related to that question.",
            [],
        )

    context_text = format_context(docs)

    chain = prompt | llm | parser
    answer = chain.invoke({"context": context_text, "question": question})

    return answer, docs


# ---------- 3. Session state for chat history ---------- #

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content, sources?)]


# ---------- 4. Sidebar: settings / controls ---------- #

with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("**Model**: `gpt-4o-mini` (via langchain-openai)")
    st.markdown("**Vector store**: `Chroma (policy)`")

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        # st.experimental_rerun() # streamlit has no attribute 'experimental_rerun'
        st.rerun()  # use this instead to rerun the app

    st.markdown("---")
    st.markdown(
        """
        **Tip**: Ask questions like:
        - *"What is the refund policy?"*
        - *"Can I cancel after 30 days?"*
        - *"What happens if payment is late?"*
        """
    )


# ---------- 5. Main layout ---------- #

st.title("💬 Policy RAG Chatbot")
st.caption("Ask questions grounded in your policy knowledge base (Chroma + OpenAI).")

chat_container = st.container()
input_container = st.container()

# Display history
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

                # Optional: show sources if present
                if "sources" in msg and msg["sources"]:
                    with st.expander("View sources used for this answer"):
                        for i, d in enumerate(msg["sources"], start=1):
                            st.markdown(f"**Source {i}**")
                            st.write(d.page_content)
                            meta = getattr(d, "metadata", None)
                            if meta:
                                st.caption(str(meta))


# Input + response
with input_container:
    user_input = st.chat_input("Ask a question about the policy...")

    if user_input:
        # 1) Add user message to history
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        # 2) Show it immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        # 3) Run RAG
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, docs = run_rag(user_input)
                st.markdown(answer)

                if docs:
                    with st.expander("View sources used for this answer"):
                        for i, d in enumerate(docs, start=1):
                            st.markdown(f"**Source {i}**")
                            st.write(d.page_content)
                            meta = getattr(d, "metadata", None)
                            if meta:
                                st.caption(str(meta))

        # 4) Save assistant message (with sources) in history
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": docs}
        )


