from __future__ import annotations

import os
import textwrap
from typing import List

import chromadb
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain.vectorstores import Chroma
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

DEFAULT_DB_PATH = "./chroma_db"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_K = 5

st.sidebar.title("⚙️ Settings")
chroma_path = st.sidebar.text_input("Chroma DB path", DEFAULT_DB_PATH)
model_name = st.sidebar.text_input("OpenAI model", DEFAULT_MODEL)
retrieval_k = st.sidebar.slider("Chunks per query", 1, 20, DEFAULT_K)
model_temp = st.sidebar.slider("LLM temperature", 0.0, 1.0, 0.0, 0.1)

@st.cache_resource(show_spinner="Loading vector store…")
def load_vector_store(path: str) -> Chroma:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Chroma directory '{path}' not found – run your ingestion script first.")

    client = chromadb.PersistentClient(path=path)
    collection = client.get_default_collection()  #for now i only have one collection
    return Chroma(
        client=client,
        collection_name=collection.name,
        embedding_function=OpenAIEmbeddings(),
    )

@st.cache_resource(show_spinner="Building RAG chain…")
def build_chain(vs: Chroma, model: str, k: int, temperature: float) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name=model, temperature=temperature, streaming=False)
    retriever = vs.as_retriever(search_kwargs={"k": k, "search_type": "mmr"})  # MMR = better diversity
    return ConversationalRetrievalChain.from_llm(
        llm,
        retriever,
        return_source_documents=True,
    )

try:
    vector_store = load_vector_store(chroma_path)
    rag_chain = build_chain(vector_store, model_name, retrieval_k, model_temp)
except Exception as err:
    st.error(f"❌ {err}")
    st.stop()


st.title("💬 Class Recording Chatbot")
st.caption("Ask anything about the processed Zoom transcripts – the assistant cites the most relevant chunks.")

# Session‑level chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Replay history so far
for msg in st.session_state.history:
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.markdown(msg.content)
        # Show context expander for assistant messages if present
        if role == "assistant" and msg.additional_kwargs.get("context_blocks"):
            with st.expander("Retrieved context"):
                for block in msg.additional_kwargs["context_blocks"]:
                    st.markdown(block)

# Chat input (returns None until user submits)
user_query = st.chat_input("Type your question and press Enter…")

if user_query:
    # Display user's message immediately
    st.session_state.history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the RAG chain with existing chat history
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = rag_chain({
                "question": user_query,
                "chat_history": st.session_state.history,
            })

            answer: str = result["answer"]
            source_docs = result["source_documents"]

            # Build context blocks for display & storage
            context_blocks: List[str] = []
            for i, doc in enumerate(source_docs, 1):
                header = f"**Chunk {i}** – {doc.metadata.get('source', 'unknown')}"
                wrapped = textwrap.fill(doc.page_content, 100)
                context_blocks.append(f"{header}\n{wrapped}")

            st.markdown(answer)
            with st.expander("Retrieved context"):
                for block in context_blocks:
                    st.markdown(block)

    # Save assistant message (including context) to history
    assistant_msg = AIMessage(content=answer)
    assistant_msg.additional_kwargs["context_blocks"] = context_blocks
    st.session_state.history.append(assistant_msg)