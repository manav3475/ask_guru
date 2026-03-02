# Ask Guru 🧙‍♂️

A **Retrieval Augmented Generation (RAG)** based chatbot designed to answer all your career-related queries by intelligently searching through your knowledge base of documents.

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)

## Overview

Ask Guru is an intelligent chatbot that combines modern NLP techniques with efficient document retrieval to provide accurate, context-aware answers to career-related questions. The system processes PDF documents, converts them into embeddings, and uses vector search to retrieve the most relevant information before generating responses.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Data Processing                         │
│  PDFs → Chunking → Embeddings → Vector Store (Qdrant)          │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                        (Query Retrieval)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
│  User → Gradio UI → FastAPI Backend → LLM → Answer              │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow:
1. **Document Ingestion**: PDF files are loaded from the data directory
2. **Chunking**: Documents are split into manageable chunks (800 tokens with 215 overlap)
3. **Embedding**: Text chunks are converted to vector embeddings using HuggingFace models
4. **Vector Storage**: Embeddings are stored in Qdrant vector database
5. **Query Processing**: User questions are embedded and matched against stored documents
6. **Answer Generation**: Retrieved documents are passed to an LLM for response generation

## 📁 Project Structure

```
ask_guru/
├── config.py              # Configuration settings and environment variables
├── main.py                # FastAPI application and API endpoints
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # This file
├── data/                  # Directory for PDF documents
├── logs/                  # Application logs
├── frontend/
│   └── UI.py              # Gradio user interface
└── src/
    ├── __init__.py
    ├── generate.py        # LLM-based answer generation
    ├── index.py           # Document indexing and embedding creation
    ├── logger.py          # Logging utilities
    └── retrieval.py       # Vector similarity search and document retrieval
```

## ✨ Features

- **PDF Processing**: Automatically processes multiple PDF documents
- **Intelligent Chunking**: Splits documents into optimal chunk sizes with overlap
- **Fast Vector Search**: Uses Qdrant for efficient similarity search
- **Web Interface**: Gradio-based user-friendly chat interface
- **RESTful API**: FastAPI backend for programmatic access
- **Configurable**: Easy-to-modify configuration for different use cases
- **Logging**: Comprehensive logging for debugging and monitoring

## 🛠️ Technologies

- **LLM & Embeddings**: 
  - LangChain for LLM orchestration
  - HuggingFace for embeddings
  - OpenAI integration for advanced LLM capabilities

- **Vector Database**: 
  - Qdrant for efficient vector similarity search

- **Backend**: 
  - FastAPI for REST API
  - Gradio for web interface

- **Document Processing**: 
  - PyPDF for PDF extraction
  - LangChain for document processing and chunking

- **Environment**: 
  - Python 3.10+
  - python-dotenv for environment management

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- Qdrant server running (or Qdrant Cloud account)
- HuggingFace API token
- OpenAI API key (optional, for using GPT models)

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ask_guru
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -e .
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env  # Create .env file with your credentials
   ```

## ⚙️ Configuration

Edit `config.py` to customize the following parameters:

```python
# Qdrant Configuration
QDRANT_URL = "your-qdrant-url"
QDRANT_API_KEY = "your-api-key"
COLLECTION_NAME = "Ask_Guru"

# Document Processing
DATA_DIR = "./data/*.pdf"
CHUNK_SIZE = 800           # Size of text chunks
CHUNK_OVERLAP = 215        # Overlap between chunks
```

### Environment Variables (.env)
```
HUGGINGFACEHUB_ACCESS_TOKEN=your_token_here
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key  # Optional
```

## 🚀 Usage

### 1. Index Documents
Process and index your PDF documents:
```bash
python -c "from src.index import index_documents; index_documents()"
```

### 2. Start the API Server
```bash
uvicorn main:app --reload
```

### 3. Launch the Web UI
```bash
python frontend/UI.py
```

### 4. API Endpoints

#### Health Check
```bash
GET /
```

#### Chat Endpoint
```bash
POST /chat
Content-Type: application/json

{
  "query": "What skills should I develop for a career in AI?",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "Based on your knowledge base, the essential skills for AI careers include..."
}
```

## 📝 Module Documentation

- **`config.py`**: Centralized configuration management
- **`main.py`**: FastAPI server with chat endpoints
- **`src/index.py`**: Document indexing and embedding pipeline
- **`src/retrieval.py`**: Vector similarity search implementation
- **`src/generate.py`**: LLM response generation
- **`src/logger.py`**: Custom logging configuration
- **`frontend/UI.py`**: Gradio web interface

## 📊 Dependencies

Key dependencies include:
- fastapi >= 0.116.1
- langchain >= 0.3.27
- qdrant-client >= 1.15.1
- gradio >= 5.44.1
- pypdf >= 6.0.0
- fastembed >= 0.7.3

For complete list, see `pyproject.toml`

## 🔍 Troubleshooting

- **Qdrant Connection Issues**: Verify QDRANT_URL and API_KEY in .env
- **Missing Dependencies**: Run `pip install -e .` again
- **PDF Processing Errors**: Ensure PDFs are in `data/` directory and not corrupted

## 🤝 Contributing

Contributions are welcome! Please follow the existing code style and add tests for new features.

## 📧 Contact

**Manav Shah**
- Email: manavshah712@gmail.com
