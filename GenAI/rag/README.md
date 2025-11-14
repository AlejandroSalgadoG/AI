# RAG System for The Batch Articles

A Retrieval-Augmented Generation (RAG) system that scrapes, processes, and answers questions about articles from [The Batch](https://www.deeplearning.ai/the-batch/), DeepLearning.AI's weekly newsletter on AI developments.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Populating the Database](#populating-the-database)
  - [Querying the RAG System](#querying-the-rag-system)
  - [Running Tests](#running-tests)
- [Project Structure](#project-structure)
- [Component Details](#component-details)
- [Development](#development)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Overview

This RAG system provides an intelligent question-answering interface for articles published in The Batch. It:

1. **Scrapes** article content from The Batch website
2. **Processes** text into meaningful chunks
3. **Stores** embeddings in a vector database (Chroma)
4. **Retrieves** relevant context for user queries
5. **Generates** accurate answers using local LLMs via Ollama

The system uses LangChain for orchestration, Chroma for vector storage, and Ollama for running local language models.

## Architecture

```
┌─────────────┐
│   Scraper   │──> Fetches articles from The Batch
└──────┬──────┘
       │
       ▼
┌─────────────┐
│TextHandler  │──> Splits text into chunks
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Storage    │──> Stores embeddings in ChromaDB
└──────┬──────┘
       │
       ▼
┌─────────────┐
│     RAG     │──> Retrieves context & generates answers
└─────────────┘
```

### Data Flow

1. **Data Ingestion**: `scraper.py` fetches articles from specified dates
2. **Text Processing**: `text_handler.py` chunks text with overlap for better context
3. **Vector Storage**: `storage.py` creates embeddings and stores them in ChromaDB
4. **Query Processing**: `rag.py` retrieves relevant chunks and generates answers
5. **Evaluation**: `test_rag.py` validates system performance using metrics

## Features

- **Web Scraping**: Automated scraping of The Batch articles by date
- **Intelligent Chunking**: Recursive text splitting with configurable overlap
- **Vector Search**: Semantic search using Chroma vector database
- **Local LLMs**: Privacy-friendly inference using Ollama
- **Retry Logic**: Fault-tolerant operations with automatic retries
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Automated Testing**: DeepEval integration for RAG metrics
- **Configurable Models**: Easy switching between different Ollama models

## Prerequisites

### System Requirements

- **Python**: 3.12.10 or higher
- **Ollama**: For running local language models
  - Download from: https://ollama.com/download

### Required Ollama Models

```bash
ollama pull qwen2.5:latest      # Primary LLM for answer generation
ollama pull llama3.2:latest     # Embedding model
ollama pull deepseek-r1:1.5b    # Testing/evaluation model
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Ollama Installation

```bash
ollama list  # Should show installed models
```

## Configuration

Configuration is managed through `config.py` using dataclasses. Three main configurations exist:

### Text Processing Configuration

```python
"text": {
    "chunk_size": 1000,          # Characters per chunk
    "chunk_overlap": 200,        # Overlap between chunks
    "add_start_index": True      # Track original position
}
```

### Storage Configuration (Embeddings)

```python
"storage": {
    "model": "llama3.2",         # Ollama embedding model
    "temperature": 0              # Deterministic embeddings
}
```

### LLM Configuration (Answer Generation)

```python
"llm": {
    "model": "qwen2.5",          # Ollama model for answers
    "temperature": 0,             # Deterministic responses
    "prompt": "..."              # System prompt template
}
```

To modify configurations, edit `config.py` directly.

## Usage

### Populating the Database

Scrape articles from The Batch and populate the vector database:

```bash
# Show help
python populate_db.py -h

# Scrape 1 issue starting from May 14, 2025 (default)
python populate_db.py

# Scrape 5 issues starting from January 1, 2025
python populate_db.py 01-01-2025 5

# Scrape 10 issues starting from March 5, 2025
python populate_db.py 05-03-2025 10
```

**Important Notes:**
- Dates must be Wednesdays (The Batch publication day)
- Date format: `dd-mm-yyyy`
- The script goes backwards in time from the start date
- Articles are stored in `data/chroma_db/`

### Querying the RAG System

Ask questions about the ingested articles:

```bash
# Using default query
python rag.py

# Custom query
python rag.py "What is phi-4-reasoning?"

# More examples
python rag.py "What are the latest developments in EU AI regulations?"
python rag.py "Explain memory layers in neural networks"
```

### Running Tests

Evaluate RAG performance using DeepEval metrics:

```bash
# Configure DeepEval to use Ollama
deepeval set-ollama deepseek-r1:1.5b

# Run evaluation
python test_rag.py
```

The test suite evaluates:
- **Answer Relevancy**: How relevant is the answer to the question?
- **Faithfulness**: Is the answer grounded in the retrieved context?
- **Contextual Precision**: How precise is the retrieved context?
- **Contextual Recall**: How complete is the retrieved context?
- **Contextual Relevancy**: How relevant is the retrieved context?

## Project Structure

```
rag/
├── config.py              # Configuration management
├── definitions.py         # Data structures (WebData)
├── failback.py           # Retry decorator for fault tolerance
├── logs.py               # Logging configuration
├── populate_db.py        # Database population script
├── rag.py                # Main RAG implementation
├── scraper.py            # Web scraping utilities
├── storage.py            # Vector database management
├── test_dataset.py       # Golden test data
├── test_rag.py           # RAG evaluation suite
├── text_handler.py       # Text chunking logic
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/
│   └── chroma_db/       # Vector database storage
└── venv/                # Virtual environment
```

## Component Details

### `scraper.py` - Web Scraping

**Key Functions:**
- `get_articles_url(date)`: Fetches article URLs for a given date
- `iter_dates(start_date, delta, n)`: Generates date range for scraping

**WebLoader Class:**
- Extends LangChain's `WebBaseLoader`
- Extracts article content from `<article>` tags
- Builds metadata (title, description, language, UUID)
- Implements retry logic for network failures

### `text_handler.py` - Text Processing

**TextHandler Class:**
- Uses `RecursiveCharacterTextSplitter` from LangChain
- Splits text into overlapping chunks for better context
- Preserves metadata across chunks
- Prepends article title to content

### `storage.py` - Vector Database

**Storage Class:**
- Manages ChromaDB vector store
- Creates embeddings using Ollama models
- Provides retriever interface for RAG
- Persists data to disk (`data/chroma_db/`)

**Methods:**
- `get_vector_store_retriver()`: Returns configured retriever
- `add_documents(docs)`: Adds documents to vector store

### `rag.py` - RAG Implementation

**Rag Class:**
- Orchestrates retrieval and generation
- Uses LangChain LCEL (LangChain Expression Language)
- Implements retry logic for robustness

**Methods:**
- `get_context(query)`: Retrieves relevant documents
- `parse_retrived(data)`: Formats retrieved context
- `consult(query)`: End-to-end RAG pipeline

**Chain Architecture:**
```python
{
    "context": retriever,
    "question": user_query
}
| prompt_template
| llm
| output_parser
```

### `failback.py` - Retry Mechanism

**retry Decorator:**
- Configurable retry attempts (default: 3)
- Logs each failure with attempt number
- Raises exception after all attempts fail
- Used for network requests and LLM calls

### `logs.py` - Logging

**Features:**
- Console and file logging
- Timestamped log files (`execution-DD-MM-YY_HH-MM-SS.log`)
- Formatted output with timestamps and log levels
- INFO level by default

### `test_rag.py` - Evaluation

**Test Suite:**
- Uses DeepEval framework
- Tests against golden dataset
- Computes multiple RAG metrics
- Validates retrieval and generation quality

## Development

### Code Quality Tools

The project includes several development tools:

```bash
# Linting with Ruff
ruff check .

# Code formatting with Black
black .

# Type checking with MyPy
mypy .

# Run all checks (if lint.sh exists)
bash lint.sh
```

### Adding New Test Cases

Edit `test_dataset.py` to add new golden examples:

```python
golden_data.append({
    "question": "Your question here?",
    "expected_answer": "Expected answer here"
})
```

### Modifying the Prompt

Edit the prompt in `config.py`:

```python
configs["llm"].prompt = """
Your custom system prompt here.
Context: {context}
Question: {question}
Answer:
"""
```

### Using Different Models

Change models in `config.py`:

```python
# For embeddings
configs["storage"].model = "nomic-embed-text"

# For generation
configs["llm"].model = "llama3.2:70b"
```

Make sure to pull new models first:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:70b
```

## Testing

### Manual Testing

```bash
# Test scraper
python scraper.py

# Test text handler
python text_handler.py

# Test end-to-end
python rag.py "test question"
```

### Automated Testing

```bash
# Run full evaluation suite
python test_rag.py

# Check logs
tail -f execution-*.log
```

### Interpreting Test Results

DeepEval provides scores (0-1) for each metric:
- **≥0.8**: Excellent
- **0.6-0.8**: Good
- **0.4-0.6**: Fair
- **<0.4**: Needs improvement

## Troubleshooting

### Common Issues

**1. Ollama Connection Error**

```
Error: Could not connect to Ollama
```

**Solution:**
```bash
# Start Ollama service
ollama serve

# In another terminal, verify
ollama list
```

**2. No Articles Found**

```
INFO: No articles found
```

**Solution:**
- Verify the date is a Wednesday
- Check internet connection
- Verify The Batch published that week
- Try a different date

**3. Import Errors**

```
ModuleNotFoundError: No module named 'langchain'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**4. ChromaDB Permission Error**

```
PermissionError: [Errno 13] Permission denied: 'data/chroma_db'
```

**Solution:**
```bash
# Create directory with proper permissions
mkdir -p data/chroma_db
chmod 755 data/chroma_db
```

**5. Model Not Found**

```
Error: Model 'qwen2.5' not found
```

**Solution:**
```bash
# Pull the required model
ollama pull qwen2.5:latest
```

### Debug Mode

Enable verbose logging by modifying `logs.py`:

```python
logging.basicConfig(
    format=log_format,
    datefmt=log_d_format,
    level=logging.DEBUG,  # Changed from INFO
)
```

### Performance Optimization

**Slow Retrieval:**
- Reduce chunk overlap in `config.py`
- Use smaller embedding models
- Limit retriever results: `extra_params={"k": 3}`

**Poor Answer Quality:**
- Increase chunk size for more context
- Use larger LLM models
- Adjust temperature (0.1-0.3 for more variability)
- Refine the system prompt

**Memory Issues:**
- Use smaller models (e.g., `qwen2.5:3b` instead of `qwen2.5:7b`)
- Process fewer articles at once
- Reduce chunk size

## Dependencies

Full list in `requirements.txt`:

- **langchain==0.3.19**: RAG orchestration framework
- **langchain-chroma==0.2.2**: ChromaDB integration
- **langchain-community==0.3.18**: Community integrations
- **langchain-ollama==0.3.3**: Ollama integration
- **langchain-text-splitters==0.3.6**: Text chunking utilities
- **beautifulsoup4==4.13.3**: HTML parsing
- **requests==2.32.3**: HTTP requests
- **ruff==0.11.10**: Python linter
- **black==25.1.0**: Code formatter
- **mypy==1.15.0**: Static type checker
- **types-requests==2.32.0.20250515**: Type stubs for requests
- **deepeval==2.9.4**: RAG evaluation framework

## References

- Articles sourced from [The Batch](https://www.deeplearning.ai/the-batch/) by DeepLearning.AI
- Built with [LangChain](https://www.langchain.com/)
- Vector database powered by [Chroma](https://www.trychroma.com/)
- Local inference via [Ollama](https://ollama.com/)
- Evaluation using [DeepEval](https://docs.confident-ai.com/)
