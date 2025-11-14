# RAG with Image Summary

A multimodal Retrieval-Augmented Generation (RAG) system that processes and queries articles from [The Batch](https://www.deeplearning.ai/the-batch/) newsletter. This system extracts both textual content and images, uses AI to generate image summaries, and stores everything in a vector database for efficient semantic search and question-answering.

## Dependencies

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Populating the Database](#populating-the-database)
  - [Querying the RAG System](#querying-the-rag-system)
  - [Testing](#testing)
- [Project Structure](#project-structure)
- [Components](#components)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Overview

This project implements a RAG pipeline that:
1. **Scrapes** articles from deeplearning.ai's "The Batch" newsletter
2. **Processes** text content by splitting it into semantic chunks
3. **Analyzes** images using multimodal AI (llava) to generate descriptive summaries
4. **Stores** both text and image summaries in a ChromaDB vector store
5. **Retrieves** relevant context based on user queries
6. **Generates** accurate answers using a large language model

The system leverages LangChain for orchestration and Ollama for local LLM execution, ensuring privacy and control over your data.

## Features

* ✨ **Multimodal Processing**: Handles both text and images from web articles
* 🤖 **Local AI Models**: Uses Ollama for running models locally without API costs
* 🔍 **Semantic Search**: ChromaDB vector store for efficient similarity search
* 📊 **Quality Testing**: Integrated DeepEval framework for RAG evaluation
* 🔄 **Retry Mechanism**: Built-in failover for robust operation
* 📝 **Comprehensive Logging**: Detailed logging for debugging and monitoring
* ⚙️ **Configurable**: Easy-to-modify configuration for different models and parameters

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Scraper                              │
│                   (The Batch Articles)                           │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌────────────────┐      ┌────────────────┐
│ Text Handler   │      │ Image Handler  │
│ (Text Splitter)│      │ (llava:13b)    │
└────────┬───────┘      └────────┬───────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────────┐
         │   Storage (ChromaDB)  │
         │  (llama3.2 embeddings)│
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │    RAG System         │
         │    (qwen2.5)          │
         └───────────────────────┘
```

## System Requirements

### External Dependencies

- **Ollama** - Local LLM runtime
  - [Download](https://ollama.com/download)
  - Supports macOS, Linux, and Windows

- **Python 3.12.10** or higher
  - [Download](https://www.python.org/downloads/release/python-31210/)

### Hardware Recommendations

- **RAM**: Minimum 16GB (32GB recommended for llava:13b)
- **Storage**: At least 50GB free space for models and data
- **CPU/GPU**: GPU recommended for faster processing (CUDA compatible for NVIDIA, Metal for Apple Silicon)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-img-summary
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Configure Ollama Models

```bash
# Install Ollama from https://ollama.com/download

# Pull required models (this may take some time)
ollama pull qwen2.5:latest      # Main RAG LLM (~4.7GB)
ollama pull llava:13b           # Image analysis (~8GB)
ollama pull llama3.2:latest     # Embeddings (~2GB)
ollama pull deepseek-r1:1.5b    # Testing LLM (~1GB)
```

## Configuration

The system is configured through `config.py` using a centralized configuration pattern:

### Configuration Structure

```python
configs = {
    "text": {...},      # Text processing settings
    "image": {...},     # Image summarization settings
    "storage": {...},   # Vector store and embeddings
    "llm": {...}        # RAG LLM settings
}
```

### Key Configuration Options

#### Text Processing (`configs["text"]`)
- **chunk_size**: Size of text chunks (default: 1000)
- **chunk_overlap**: Overlap between chunks (default: 200)
- **add_start_index**: Track original position (default: True)

#### Image Analysis (`configs["image"]`)
- **model**: llava:13b (multimodal vision-language model)
- **prompt**: Instructions for image summarization

#### Storage (`configs["storage"]`)
- **model**: llama3.2 (for generating embeddings)
- **temperature**: 0 (deterministic embeddings)

#### RAG LLM (`configs["llm"]`)
- **model**: qwen2.5 (for answer generation)
- **temperature**: 0 (deterministic responses)
- **prompt**: Custom prompt template for RAG

### Customizing Models

To use different models, edit `config.py`:

```python
configs["llm"] = Config(
    model="llama3.1",  # Change to your preferred model
    model_params={
        "temperature": 0.7,  # Adjust temperature
    },
    ...
)
```

## Usage

### Populating the Database

The `populate_db.py` script scrapes articles from The Batch and stores them in the vector database.

#### Basic Usage

```bash
python populate_db.py
```

This uses default values:
- Start date: 14-05-2025 (Wednesday)
- Number of issues: 1

#### Advanced Usage

```bash
python populate_db.py <start_date> <num_issues>
```

**Parameters:**
- `start_date`: Wednesday date in format `dd-mm-yyyy`
- `num_issues`: Number of weekly issues to scrape (going backwards from start_date)

**Examples:**

```bash
# Scrape 1 issue from May 14, 2025
python populate_db.py 14-05-2025 1

# Scrape 5 issues starting from March 5, 2025
python populate_db.py 05-03-2025 5

# Scrape 10 weeks of articles
python populate_db.py 14-05-2025 10
```

#### What Happens During Population

1. **Date Calculation**: Generates list of Wednesday dates
2. **URL Discovery**: Finds article URLs for each date
3. **Web Scraping**: Downloads article content and images
4. **Text Processing**: Splits text into semantic chunks
5. **Image Analysis**: Generates AI summaries of images
6. **Storage**: Stores embeddings in ChromaDB

**Output:**
- Vector database: `data/chroma_db/`
- Execution log: `execution-<timestamp>.log`

### Querying the RAG System

Once the database is populated, you can query it using `rag.py`.

#### Basic Query

```bash
python rag.py "<your question>"
```

#### Examples

```bash
# Query about specific topics
python rag.py "What do you know about Microsoft phi-4-reasoning?"

# Query without argument (uses default question)
python rag.py

# Query about images
python rag.py "What visualizations are shown in recent articles?"

# Complex queries
python rag.py "How are memory layers different from attention mechanisms?"
```

#### How RAG Works

1. **Query Processing**: Your question is embedded into a vector
2. **Retrieval**: Finds most relevant chunks from the database
3. **Context Building**: Assembles relevant text and image summaries
4. **Generation**: LLM generates answer based on retrieved context
5. **Response**: Returns accurate, context-aware answer

#### Programmatic Usage

```python
from rag import Rag

# Initialize RAG system
rag = Rag()

# Query the system
answer = rag.consult("What are the latest AI trends?")
print(answer)

# Access retrieved context
print(rag.last_context)
```

### Testing

The project includes a comprehensive testing suite using DeepEval.

#### Setup Testing

```bash
# Configure DeepEval to use local model
deepeval set-ollama deepseek-r1:1.5b
```

#### Run Tests

```bash
python test_rag.py
```

#### Test Metrics

The testing framework evaluates:

1. **Answer Relevancy**: Does the answer address the question?
2. **Faithfulness**: Is the answer grounded in the retrieved context?
3. **Contextual Precision**: Are relevant documents ranked higher?
4. **Contextual Recall**: Are all relevant documents retrieved?
5. **Contextual Relevancy**: Is the retrieved context relevant to the question?

#### Test Dataset

Tests are defined in `test_dataset.py` with golden questions and expected answers:

```python
golden_data = [
    {
        "question": "What is phi-4-reasoning?",
        "expected_answer": "Phi-4-reasoning is a fine-tuned version..."
    },
    ...
]
```

#### Adding Custom Tests

Edit `test_dataset.py`:

```python
golden_data.append({
    "question": "Your new question?",
    "expected_answer": "Expected answer based on your data"
})
```

## Project Structure

```
rag-img-summary/
├── config.py              # Central configuration
├── definitions.py         # Data classes (WebData, WebImage)
├── failback.py           # Retry decorator for reliability
├── logs.py               # Logging configuration
├── scraper.py            # Web scraping logic
├── text_handler.py       # Text processing and chunking
├── image_handler.py      # Image analysis with llava
├── storage.py            # ChromaDB vector store management
├── populate_db.py        # Database population script
├── rag.py                # RAG query system
├── test_dataset.py       # Test cases definition
├── test_rag.py           # Testing script
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── data/                 # Data directory
│   └── chroma_db/        # ChromaDB storage
└── venv/                 # Virtual environment
```

## Components

### 1. Scraper (`scraper.py`)

**Purpose**: Scrapes articles from deeplearning.ai's The Batch newsletter

**Key Classes:**
- `WebLoader`: Custom web scraper extending LangChain's WebBaseLoader

**Features:**
- Finds articles by date
- Extracts text from `<article>` tags
- Downloads and encodes images
- Builds comprehensive metadata

**Usage:**
```python
from scraper import WebLoader, get_articles_url
from datetime import date

articles = get_articles_url(date(2025, 5, 14))
loader = WebLoader(web_paths=articles)
web_data = loader.load_web_paths()
```

### 2. Text Handler (`text_handler.py`)

**Purpose**: Processes text content into semantic chunks

**Key Classes:**
- `TextHandler`: Manages text splitting and document creation

**Features:**
- Uses LangChain's RecursiveCharacterTextSplitter
- Configurable chunk size and overlap
- Preserves metadata across chunks
- Adds unique UUIDs to each chunk

**Configuration:**
```python
configs["text"].extra_params = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "add_start_index": True,
}
```

### 3. Image Handler (`image_handler.py`)

**Purpose**: Generates AI summaries of images

**Key Classes:**
- `ImageHandler`: Manages image analysis with multimodal LLM

**Features:**
- Uses llava:13b vision-language model
- Generates retrieval-optimized summaries
- Handles base64-encoded images
- Includes retry logic for reliability

**Process:**
1. Receives base64-encoded image
2. Sends to llava with summarization prompt
3. Returns concise, searchable description

### 4. Storage (`storage.py`)

**Purpose**: Manages ChromaDB vector store

**Key Classes:**
- `Storage`: Wrapper for ChromaDB operations

**Features:**
- Persistent storage in `data/chroma_db/`
- Ollama embeddings (llama3.2)
- Collection management
- Retriever configuration

**Methods:**
- `add_documents()`: Add documents to vector store
- `get_vector_store_retriver()`: Get configured retriever
- `store()`: Store with label conversion

### 5. RAG System (`rag.py`)

**Purpose**: Main RAG query interface

**Key Classes:**
- `Rag`: Complete RAG pipeline

**Features:**
- LangChain LCEL pipeline
- Context retrieval from vector store
- LLM-based answer generation
- Context tracking for testing

**Pipeline:**
```python
{context, question} → prompt → LLM → answer
```

### 6. Utilities

#### Definitions (`definitions.py`)
- `WebImage`: Image data structure
- `WebData`: Web page data structure

#### Failback (`failback.py`)
- `@retry`: Decorator for automatic retries (default: 3 attempts)

#### Logs (`logs.py`)
- Configured logging to console and file
- Timestamped log files: `execution-<timestamp>.log`

## Examples

### Example 1: Complete Workflow

```bash
# 1. Populate database with 3 weeks of articles
python populate_db.py 14-05-2025 3

# 2. Query the system
python rag.py "What are the latest developments in AI reasoning?"

# 3. Run tests
python test_rag.py
```

### Example 2: Custom Scraping

```python
from scraper import WebLoader
from text_handler import TextHandler
from image_handler import ImageHandler
from storage import Storage

# Define custom articles
articles = [
    "https://www.deeplearning.ai/the-batch/issue-..."
]

# Initialize handlers
loader = WebLoader(web_paths=articles)
text_handler = TextHandler()
image_handler = ImageHandler()
storage = Storage()

# Process each article
for web_data in loader.lazy_load_web_paths():
    # Process text
    text_docs = text_handler.apply(web_data)

    # Process images
    image_docs = image_handler.apply(web_data)

    # Store everything
    storage.store(text_docs + image_docs)
```

### Example 3: Custom RAG Query

```python
from rag import Rag

# Initialize
rag = Rag()

# Ask multiple questions
questions = [
    "What is phi-4-reasoning?",
    "How do memory layers work?",
    "What are the latest AI regulations?"
]

for question in questions:
    print(f"\nQ: {question}")
    answer = rag.consult(question)
    print(f"A: {answer}")
    print(f"Context sources: {len(rag.last_context)}")
```

### Example 4: Batch Processing

```python
from datetime import date, timedelta
from scraper import iter_dates, get_articles_url, WebLoader
from text_handler import TextHandler
from image_handler import ImageHandler
from storage import Storage

# Initialize
storage = Storage()
text_handler = TextHandler()
image_handler = ImageHandler()

# Process multiple weeks
dates = iter_dates(
    start_date=date(2025, 5, 14),
    delta=timedelta(days=7),
    n=5  # 5 weeks
)

total_docs = 0
for d in dates:
    print(f"Processing: {d}")

    if articles := get_articles_url(d):
        loader = WebLoader(web_paths=articles)

        for web_data in loader.lazy_load_web_paths():
            text_docs = text_handler.apply(web_data)
            image_docs = image_handler.apply(web_data)

            all_docs = text_docs + image_docs
            storage.store(all_docs)
            total_docs += len(all_docs)

print(f"Total documents stored: {total_docs}")
```

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Error

**Error**: `Connection refused` or `Model not found`

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama service (if needed)
ollama serve

# Verify models are pulled
ollama pull qwen2.5:latest
ollama pull llava:13b
ollama pull llama3.2:latest
```

#### 2. Memory Issues with llava:13b

**Error**: `Out of memory` when processing images

**Solutions**:
- Use smaller model: Change `configs["image"].model = "llava:7b"`
- Process fewer images at once
- Close other applications
- Increase system swap space

#### 3. No Articles Found

**Error**: "No articles found" when running populate_db.py

**Solutions**:
- Verify date is a Wednesday
- Check internet connection
- Verify The Batch publishes on that date
- Try a different date range

#### 4. ChromaDB Errors

**Error**: Database lock or corruption issues

**Solutions**:
```bash
# Remove and rebuild database
rm -rf data/chroma_db/
python populate_db.py
```

#### 5. Import Errors

**Error**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 6. Slow Performance

**Issues**: Queries take too long

**Solutions**:
- Use GPU acceleration (check Ollama GPU support)
- Use smaller models (e.g., qwen2.5:0.5b instead of qwen2.5:latest)
- Reduce chunk size in config.py
- Limit retrieval results in storage config

### Debug Mode

Enable detailed logging:

```python
# In logs.py, change level
logging.basicConfig(
    format=log_format,
    datefmt=log_d_format,
    level=logging.DEBUG,  # Changed from INFO
)
```

### Performance Tips

1. **Model Selection**: Smaller models are faster but less accurate
2. **Batch Processing**: Process multiple articles in one session
3. **GPU Usage**: Ensure Ollama uses GPU if available
4. **Chunk Size**: Larger chunks mean fewer embeddings but less precision
5. **Embedding Cache**: ChromaDB caches embeddings automatically

## Development Setup

```bash
# Install development tools
pip install ruff black mypy

# Run linter
ruff check .

# Run formatter
black .

# Run type checker
mypy .
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings to functions and classes
- Keep functions focused and small
- Write descriptive commit messages

### Testing Guidelines

- Add tests for new features in `test_dataset.py`
- Ensure all tests pass before committing
- Maintain test coverage above 80%

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linters
5. Submit pull request with description

## Refences

- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM runtime
- **ChromaDB**: Vector database
- **DeepEval**: LLM evaluation framework
- **The Batch**: deeplearning.ai's newsletter (data source)
