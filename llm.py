import streamlit as st
import openai
__import__('pysqlite3-binary')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3-binary')
import chromadb
import os
from chromadb.config import Settings

# -------------------------------------
# CONFIGURATION
# -------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Initialize Chroma persistent client with local persistence in "chroma_db" folder
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="srt-subtitles")

# -------------------------------------
# HELPER FUNCTIONS
# -------------------------------------
def get_embedding(text):
    """Generates an embedding for the provided text using OpenAI."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def retrieve_context(query, top_k=5):
    """Retrieves context documents from Chroma based on the query embedding."""
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    # Aggregate the retrieved documents as context.
    context_docs = results.get("documents", [[]])[0]
    context = "\n".join(context_docs)
    return context

def answer_query(query, context):
    """Generates an answer using OpenAI Chat API with the provided context."""
    prompt = f"""You are a helpful assistant knowledgeable about class recordings.
Given the following context extracted from class recordings, answer the question below.

Context:
{context}

Question: {query}

Answer:"""
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant knowledgeable about class recordings."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    answer = response.choices[0].message.content.strip()
    return answer

# -------------------------------------
# STREAMLIT APP
# -------------------------------------
def main():
    st.title("Class Recording Q&A")
    st.write("Ask questions based on the processed class recording data.")

    query = st.text_input("Enter your question:")
    if st.button("Get Answer") and query:
        with st.spinner("Retrieving context and generating an answer..."):
            context = retrieve_context(query)
            answer = answer_query(query, context)
            st.markdown("### Answer")
            st.write(answer)
            st.markdown("### Retrieved Context")
            st.write(context)

if __name__ == "__main__":
    main()