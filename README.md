#  Document Q&A using Local RAG (Ollama)

A **local Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF document and ask questions about its content. The system retrieves relevant document chunks using vector similarity search and generates accurate answers using a **locally hosted LLM via Ollama**.

This project demonstrates a complete **end-to-end RAG pipeline** built with **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Streamlit**.

---

## Features

* 📤 Upload any PDF document
* ✂️ Intelligent text chunking with overlap
* 🧠 Semantic search using vector embeddings (FAISS)
* 🤖 Local LLM inference using Ollama (no API calls)
* 🔒 Answers strictly grounded in document context
* 🖥️ Simple and interactive Streamlit UI

---

## 🛠️ Tech Stack

* **Frontend / UI**: Streamlit
* **LLM Framework**: LangChain
* **Vector Store**: FAISS
* **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
* **LLM**: `gemma:2b` (served locally via Ollama)
* **Document Loader**: PyPDFLoader

---

## 🧩 Architecture Overview

```
PDF Upload
   ↓
Document Loader (PyPDFLoader)
   ↓
Text Splitter (RecursiveCharacterTextSplitter)
   ↓
Embeddings (HuggingFace)
   ↓
Vector Store (FAISS)
   ↓
Retriever (Top-K Similarity Search)
   ↓
Prompt + Context Injection
   ↓
Local LLM (Ollama)
   ↓
Answer
```


## 🦙 Ollama Setup (Required)

Install Ollama from:
👉 [https://ollama.com](https://ollama.com)

Pull the required model:

```bash
ollama pull gemma:2b
```

Make sure Ollama is running in the background before starting the app.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. User uploads a PDF file
2. PDF text is extracted and split into overlapping chunks
3. Each chunk is converted into vector embeddings
4. FAISS stores and indexes embeddings for fast similarity search
5. User asks a question
6. Relevant chunks are retrieved and injected into the prompt
7. Local LLM generates an answer **only from retrieved context**

---

## 🔐 Prompt Safety

The prompt enforces **context-only answering**, reducing hallucinations:

```
Use ONLY the context below to answer.
```

This ensures responses are grounded in the uploaded document.

---

## 📌 Example Use Cases

* Internal document Q&A
* Research paper exploration
* Study notes assistant
* Offline / private document analysis
* RAG learning & experimentation

## 📈 Possible Improvements

* Add chat history (conversational memory)
* Support multiple PDFs
* Metadata-based filtering
* Better chunking strategies
* Hybrid search (BM25 + vectors)
