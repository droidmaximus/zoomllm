import os
import glob
import srt
import openai
import chromadb
import streamlit as st

# -------------------------------------
# CONFIGURATION: Set your API key and Chroma settings
# -------------------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY

# Initialize Chroma client with local persistence (DuckDB + Parquet)
client = chromadb.PersistentClient(path="chroma_db")  # Data saved in "chroma_db" folder

# Get or create the collection for SRT subtitles
collection = client.get_or_create_collection(name="srt-subtitles")

# -------------------------------------
# FUNCTIONS
# -------------------------------------
def parse_srt_file(file_path):
    """Parses an SRT file and returns a list of subtitle objects."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    subtitles = list(srt.parse(content))
    return subtitles

def get_embedding(text):
    """Generates embedding for the text using OpenAI API."""
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

def process_srt_file(file_path):
    """Processes a single SRT file: parses content, generates embeddings, and upserts to Chroma DB."""
    subtitles = parse_srt_file(file_path)
    ids, embeddings, documents, metadatas = [], [], [], []
    
    for subtitle in subtitles:
        text = subtitle.content.strip()
        if text:
            # Create a unique ID per subtitle line
            vector_id = f"{os.path.basename(file_path)}_{subtitle.index}"
            embedding = get_embedding(text)
            metadata = {
                "file": os.path.basename(file_path),
                "start": str(subtitle.start),
                "end": str(subtitle.end)
            }
            
            ids.append(vector_id)
            embeddings.append(embedding)
            documents.append(text)
            metadatas.append(metadata)
    
    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Upserted {len(ids)} vectors from {file_path}")
    else:
        print(f"No valid subtitles found in {file_path}")

def process_srt_folder(folder_path):
    """Scans the provided folder for SRT files and processes each."""
    srt_files = glob.glob(os.path.join(folder_path, '*.srt'))
    if not srt_files:
        print("No SRT files found in the folder.")
    for file_path in srt_files:
        process_srt_file(file_path)

# -------------------------------------
# MAIN EXECUTION
# -------------------------------------
if __name__ == "__main__":
    # Update folder_path to the location of your SRT files
    folder_path = r"c:\Users\ASUS\Documents\Zoom\srt_files"
    process_srt_folder(folder_path)