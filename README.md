# RAG Pipeline with LangChain and ChromaDB

A production-ready Retrieval Augmented Generation (RAG) pipeline built with FastAPI, LangChain, ChromaDB and OpenAI.

## What it does
Upload any text document and ask questions about it. The system splits the document into chunks, converts them to vectors, stores them in ChromaDB, and uses GPT-4o-mini to generate answers based on the most relevant chunks.

## Architecture

```
User
  │
  ▼
FastAPI (main.py)
  │
  ├── POST /upload ──► TextLoader ──► TextSplitter ──► OpenAIEmbeddings ──► ChromaDB
  │
  └── POST /ask ──► ChromaDB (similarity search) ──► GPT-4o-mini ──► Answer
```

## Tech Stack
- **FastAPI** — API framework
- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — Vector database for storing embeddings
- **OpenAI** — Embeddings (text-embedding-ada-002) and chat (GPT-4o-mini)
- **pytest** — Testing with mocks for external APIs

## Project Structure

```
03-rag-pipeline/
├── app/
│   ├── __init__.py
│   └── main.py        # FastAPI endpoints
│   └── rag.py         # RAG pipeline logic
│   └── sample.txt     # Sample financial document
├── tests/
│   ├── __init__.py
│   └── test_rag.py    # Unit tests with mocks
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Run the API:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload a text document |
| POST | /ask | Ask a question about the document |

## Example Usage

Upload a document:
```
POST /upload
file: sample.txt
```

Ask a question:
```
POST /ask?question=What is the role of RBA?

Response:
{
  "question": "What is the role of RBA?",
  "answer": "The Reserve Bank of Australia (RBA) is the central bank responsible for monetary policy..."
}
```

## How RAG Works

1. Document is loaded and split into 500-character chunks with 50-character overlap
2. Each chunk is converted to a vector using OpenAI embeddings
3. Vectors are stored in ChromaDB
4. When a question arrives, it is converted to a vector
5. ChromaDB finds the 3 most similar chunks using cosine similarity
6. Those chunks are sent to GPT-4o-mini with the question
7. GPT-4o-mini generates an answer based only on the provided chunks

## Tests

```bash
pytest tests/ -v
```

- Uses mocks for OpenAI and ChromaDB — no API key needed for testing
- 2/2 tests passing

---

## Türkçe Açıklama

FastAPI, LangChain, ChromaDB ve OpenAI kullanilarak olusturulmus, production'a hazir bir RAG pipeline.

## Ne Yapar?
Herhangi bir metin belgesi yukleyin ve hakkinda sorular sorun. Sistem belgeyi parcalara boler, vektore ceviri, ChromaDB'de saklar ve en alakali parcalara dayanarak GPT-4o-mini ile cevap uretir.

## RAG Nasil Calisir?

1. Belge yuklenir ve 500 karakterlik parcalara bolunur
2. Her parca OpenAI embedding modeli ile vektore cevirilir
3. Vektorler ChromaDB'ye kaydedilir
4. Soru gelince o da vektore cevirilir
5. ChromaDB cosine similarity ile en yakin 3 parcayi bulur
6. O parcalar soru ile birlikte GPT-4o-mini'ye gonderilir
7. GPT-4o-mini sadece o parcalara bakarak cevap uretir

## Kurulum

```bash
pip install -r requirements.txt
```

`.env` dosyasi olusturun:
```
OPENAI_API_KEY=your_key_here
```

API'yi baslatin:
```bash
uvicorn app.main:app --reload
```

## Testler

```bash
pytest tests/ -v
```

- OpenAI ve ChromaDB mock'lanmistir, test icin API key gerekmez
- 2/2 test gecti