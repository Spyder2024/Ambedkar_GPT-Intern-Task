# AmbedkarGPT-Intern-Task

This repository contains my solution for the Kalpit Pvt Ltd AI Intern Hiring Assignment (Phase 1). It implements a local, command-line Q&A system using a Retrieval-Augmented Generation (RAG) pipeline. The system answers questions solely from the provided `speech.txt` (an excerpt from Dr. B. R. Ambedkar’s writing) — no internet or external APIs are used.

## Problem Statement (paraphrased)

Build a simple CLI-based Q&A system that:
- Loads the provided `speech.txt`.
- Splits text into manageable chunks.
- Creates local embeddings and stores them in a local vector database.
- Retrieves relevant chunks for a user’s question.
- Generates an answer using the retrieved context and a local LLM.

Constraints and choices required by the brief:
- Language: Python 3.8+
- Orchestration: LangChain
- Vector store: ChromaDB (local)
- Embeddings: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (local; no keys)
- LLM: Ollama with Mistral 7B (local; no accounts/keys)
- Deliverables: code (`main.py`), `requirements.txt`, `README.md`, and `speech.txt` in a public repo

## Tech Stack

- Language: Python 3.8+
- Framework: LangChain (1.x module split)
- LLM: Ollama with Mistral 7B (local)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local)
- Vector store: ChromaDB (local)

## How It Works (RAG Flow)

- Document loading: `TextLoader` reads `speech.txt`.
- Chunking: `CharacterTextSplitter` creates overlapping chunks.
- Embedding: HuggingFace model produces vector representations locally.
- Indexing: Chroma persists embeddings to `./chroma_db`.
- Retrieval + Generation: Top matches are concatenated into context and passed to Ollama (Mistral) with a compact prompt to produce an answer.

## Project Structure

- `main.py` — end-to-end RAG pipeline and CLI loop
- `speech.txt` — the source document used for answering
- `requirements.txt` — Python dependencies
- `chroma_db/` — persisted local vector store (auto-created)

## Setup

Below are Windows-focused steps (PowerShell). macOS/Linux equivalents are shown afterward.

### 1) Install Ollama and pull the model

- Windows: download and install from https://ollama.ai/ (launch the app so the service runs)
- Pull the model:

```powershell
ollama pull mistral
```

Optional, to run the server explicitly:

```powershell
ollama serve
```

### 2) Python environment and dependencies (Windows PowerShell)

Use the bundled virtual environment (`gptintern`) or create your own.

- Activate the existing venv:

```powershell
& "D:Path/Ambedkar_GPT Intern Task/gptintern/Scripts/Activate.ps1"
```

- Or create a fresh one:

```powershell
python -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
```

- Install dependencies:

```powershell
python -m pip install -r requirements.txt
```


## Run

With your environment active and Ollama running:

```powershell
python main.py
```

Then type your question at the prompt, or type `exit` to quit.

## Troubleshooting

- ModuleNotFoundError for `langchain.document_loaders` or `langchain.text_splitters`:
    - LangChain 1.x split modules. This project uses:
        - `langchain_community.document_loaders`
        - `langchain_text_splitters`
        - `langchain_community.embeddings` / `vectorstores` / `llms`
    - Ensure dependencies are installed:

```powershell
python -m pip install langchain langchain-community langchain-text-splitters chromadb sentence-transformers
```

- Ollama connection issues:
    - Make sure the service is running and `mistral` is pulled.
    - Try `ollama run mistral` once to confirm the model executes.

- First run is slow:
    - Embedding the corpus and creating the Chroma index happens only once and is cached under `./chroma_db`.

## Notes on Implementation

- The classic `RetrievalQA` chain referenced in the brief isn’t present in recent LangChain 1.x. This repo composes the chain using `langchain_core` runnables (`ChatPromptTemplate` → `Ollama` → `StrOutputParser`) with a retriever from Chroma.
- Everything runs fully offline once models are pulled (Ollama + HF embeddings).

## Deliverables (as required)

- Public repository with:
    - `main.py` (well-commented)
    - `requirements.txt`
    - `README.md` (this file)
    - `speech.txt`

## License


For hiring evaluation purposes only.

