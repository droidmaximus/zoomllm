# ZoomLLM - Class Recording Q&A

A retrieval-augmented generation (RAG) application that allows you to ask questions about your class recordings. The app processes SRT subtitle files, stores embeddings in a local Chroma vector database, and uses OpenAI to generate answers based on the retrieved context.

---

## Features

- Parse and process SRT subtitle files from class recordings
- Generate text embeddings using OpenAI's text-embedding-ada-002 model
- Store embeddings locally with Chroma (persistent vector database)
- Ask natural language questions about your class content
- Streamlit-based web interface for easy interaction

---

## Project Structure

```
zoomllm/
├── main.py            # Script to process SRT files and populate the vector database
├── llm.py             # Streamlit app for Q&A interface
├── app.py             # (Optional) Additional app logic
├── requirements.txt   # Python dependencies
├── chroma_db/         # Local Chroma database (auto-generated)
└── srt_files/         # Folder containing your SRT subtitle files
```

---

## Requirements

- Python 3.10 or higher
- OpenAI API key

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/droidmaximus/zoomllm.git
   cd zoomllm
   ```
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Set up your OpenAI API key:

   - For local development, create a `.streamlit/secrets.toml` file:

     ```toml
     OPENAI_API_KEY = "your-openai-api-key-here"
     ```
   - For Streamlit Community Cloud, add the secret in the app dashboard under "Secrets".

---

## Usage

### Step 1: Process SRT Files

Place your SRT subtitle files in the `srt_files/` folder, then run:

```bash
python main.py
```

This will:

- Parse each SRT file
- Generate embeddings for each subtitle line
- Store the embeddings in the local Chroma database (`chroma_db/`)

### Step 2: Run the Q&A App

Start the Streamlit app:

```bash
streamlit run llm.py
```

Open the provided URL in your browser, enter a question about your class recordings, and get an answer based on the stored content.

---

## Deployment

### Streamlit Community Cloud

1. Push your code to GitHub (ensure `.streamlit/secrets.toml` is in `.gitignore`).
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Select your repository and deploy.
4. Add your `OPENAI_API_KEY` in the "Secrets" section of the app settings.

---

## Configuration

- **OpenAI Model**: The app uses `gpt-4.1-mini` for chat completions and `text-embedding-ada-002` for embeddings. You can change these in `llm.py` and `main.py`.
- **Chroma Database Path**: By default, the database is stored in `chroma_db/`. Modify the `path` parameter in `chromadb.PersistentClient()` to change this.

---

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`.
- **API Key Errors**: Verify your OpenAI API key is correctly set in `.streamlit/secrets.toml` or as an environment variable.
- **Protobuf Errors**: The `requirements.txt` pins `protobuf==3.20.3` to avoid compatibility issues with Chroma and OpenTelemetry.

---

## Author

droidmaximus
