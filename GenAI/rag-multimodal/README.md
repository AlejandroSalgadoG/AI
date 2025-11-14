# Multimodal RAG System

A Retrieval-Augmented Generation (RAG) system that processes and queries both text and images from [The Batch](https://www.deeplearning.ai/the-batch/) newsletter by DeepLearning.AI. This system combines multimodal embeddings, vector storage, and local LLMs to provide intelligent question-answering capabilities over scraped articles.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Populating the Database](#populating-the-database)
  - [Querying the RAG System](#querying-the-rag-system)
  - [Using the Web UI](#using-the-web-ui)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Development Tools](#development-tools)
- [Troubleshooting](#troubleshooting)

## Features

- **Multimodal Processing**: Handles both text content and images from web articles
- **Intelligent Scraping**: Automated web scraping from The Batch newsletter with date-based filtering
- **Image Summarization**: Uses vision-language models (LLaVA) to generate searchable summaries of images
- **Vector Storage**: Utilizes ChromaDB for efficient similarity search with Ollama embeddings
- **Multi-Vector Retrieval**: Employs LangChain's MultiVectorRetriever for sophisticated document retrieval
- **Local LLM Inference**: Runs entirely on local models via Ollama (no API keys required)
- **Interactive UI**: Streamlit-based chat interface for easy querying
- **Robust Error Handling**: Automatic retry mechanisms for API calls and scraping operations
- **Evaluation Framework**: Built-in testing with DeepEval metrics for RAG quality assessment

## Architecture

The system follows a modular architecture:

```
┌─────────────┐
│   Scraper   │ ──> Extracts text & images from The Batch articles
└──────┬──────┘
       │
       ├──> ┌──────────────┐
       │    │ Text Handler │ ──> Splits text into chunks
       │    └──────┬───────┘
       │           │
       └──> ┌──────────────┐
            │Image Handler │ ──> Generates image summaries using LLaVA
            └──────┬───────┘
                   │
                   ├──> ┌─────────────┐
                   │    │  ChromaDB   │ (Vector embeddings)
                   │    └─────────────┘
                   │
                   └──> ┌─────────────┐
                        │   DocStore  │ (Raw documents)
                        └─────────────┘
                              │
                              ▼
                        ┌──────────┐
                        │   RAG    │ ──> Retrieves & generates answers
                        └──────────┘
```

### Key Components

1. **Scraper** (`scraper.py`): Web scraping module that extracts articles from The Batch
2. **Text Handler** (`text_handler.py`): Processes and chunks text content
3. **Image Handler** (`image_handler.py`): Generates searchable summaries of images
4. **Storage** (`storage.py`): Manages vector store and document store
5. **RAG** (`rag.py`): Core retrieval and generation logic
6. **UI** (`ui.py`): Streamlit interface for user interaction

## Prerequisites

### System Requirements

- **Python**: 3.12.10 or higher
- **Ollama**: Latest version ([download here](https://ollama.com/download))
- **Operating System**: macOS, Linux, or Windows (with WSL recommended)
- **Memory**: 16GB RAM minimum (32GB recommended for LLaVA 13B)
- **Storage**: ~30GB free space for models and data

### Required Ollama Models

The following models must be pulled before running the system:

```bash
ollama pull llama3.2          # For embeddings
ollama pull llava:13b         # For image understanding
ollama pull gemma3            # For text generation
ollama pull deepseek-r1:1.5b  # For evaluation
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-multimodal
```

### 2. Create Virtual Environment

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Setup Ollama

1. Download and install Ollama from [ollama.com](https://ollama.com/download)
2. Start the Ollama service (usually starts automatically)
3. Pull the required models:

```bash
ollama pull llama3.2
ollama pull llava:13b
ollama pull gemma3
ollama pull deepseek-r1:1.5b
```

### 5. Verify Installation

Check that Ollama is running:

```bash
ollama list
```

You should see all four models listed.

## Configuration

The system configuration is defined in `config.py`. Key settings include:

### Model Configuration

```python
configs = {
    "text": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    "image": {
        "model": "llava:13b",
        "prompt": "Summarize this image for retrieval..."
    },
    "storage": {
        "model": "llama3.2",
        "temperature": 0,
    },
    "llm": {
        "model": "gemma3",
        "temperature": 0,
    }
}
```

### Customization Options

- **Change models**: Edit the `model` field in `config.py` for any component
- **Adjust chunking**: Modify `chunk_size` and `chunk_overlap` in the text config
- **Temperature**: Adjust `temperature` parameter for more creative/deterministic outputs
- **Prompts**: Customize system prompts for image summarization and LLM responses

## Usage

### Populating the Database

Before querying, you need to scrape and index articles from The Batch:

```bash
python populate_db.py <start_date> <num_issues>
```

#### Arguments

- `start_date` (optional): Wednesday date in format `dd-mm-yyyy` (default: `14-05-2025`)
- `num_issues` (optional): Number of issues to scrape going backwards (default: `1`)

#### Examples

```bash
# Scrape 1 issue from May 14, 2025
python populate_db.py

# Scrape 5 issues starting from February 26, 2025
python populate_db.py 26-02-2025 5

# View help
python populate_db.py -h
```

#### What Happens

1. Scrapes article URLs from The Batch for specified dates
2. Extracts text content and images from each article
3. Generates image summaries using LLaVA
4. Chunks text into searchable segments
5. Stores embeddings in ChromaDB (`data/chroma_db`)
6. Stores raw documents in local file store (`data/doc_store`)

**Note**: The start date must be a Wednesday (when The Batch is published).

### Querying the RAG System

#### Command Line Mode

Run queries directly from the terminal:

```bash
python rag.py "<your question>"
```

#### Examples

```bash
# Query about a specific topic
python rag.py "What do you know about Microsoft Phi-4-reasoning?"

# Query about recent developments
python rag.py "What are the latest advances in computer vision?"

# Use default query (if no argument provided)
python rag.py
```

#### How It Works

1. Converts your question into embeddings
2. Retrieves relevant text chunks and images from the vector store
3. Passes context and images to the LLM (Gemma3)
4. Generates a comprehensive answer based on retrieved context

### Using the Web UI

Launch the interactive Streamlit interface:

```bash
streamlit run ui.py
```

This will open a web browser with a chat interface where you can:

- Ask questions in natural language
- View conversation history
- Get real-time responses from the RAG system

The UI automatically initializes the RAG system and caches it for performance.

## Testing

The project includes a comprehensive evaluation framework using DeepEval:

### Setup DeepEval

Configure DeepEval to use your local Ollama model:

```bash
deepeval set-ollama deepseek-r1:1.5b
```

### Run Tests

```bash
python test_rag.py
```

### Evaluation Metrics

The test suite evaluates the following metrics:

1. **Answer Relevancy**: How relevant the answer is to the question
2. **Faithfulness**: Whether the answer is grounded in the retrieved context
3. **Contextual Precision**: Precision of retrieved context
4. **Contextual Recall**: Recall of retrieved context
5. **Contextual Relevancy**: Overall relevance of retrieved context

### Custom Test Data

Edit `test_dataset.py` to add your own test cases:

```python
golden_data = [
    {
        "question": "Your question here",
        "expected_answer": "Expected answer"
    }
]
```

## Project Structure

```
rag-multimodal/
├── config.py              # Configuration for all models and components
├── definitions.py         # Data classes (WebData, WebImage)
├── failback.py           # Retry decorator for error handling
├── image_handler.py      # Image processing and summarization
├── text_handler.py       # Text chunking and processing
├── scraper.py            # Web scraping logic for The Batch
├── storage.py            # Vector store and document store management
├── rag.py                # Core RAG implementation
├── populate_db.py        # Script to populate the database
├── ui.py                 # Streamlit web interface
├── logs.py               # Logging configuration
├── test_rag.py           # Evaluation script
├── test_dataset.py       # Test data for evaluation
├── requirements.txt      # Python dependencies
├── lint.sh               # Linting script
├── data/                 # Data directory (created on first run)
│   ├── chroma_db/       # Vector database
│   └── doc_store/       # Document storage
└── venv/                 # Virtual environment
```

## Technical Details

### Text Processing

- Uses `RecursiveCharacterTextSplitter` from LangChain
- Default chunk size: 1000 characters
- Chunk overlap: 200 characters
- Preserves document metadata across chunks

### Image Processing

- Encodes images in base64 format
- Generates summaries using LLaVA 13B vision-language model
- Stores both original image and summary in document store
- Retrieves images alongside text for multimodal context

### Vector Storage

- **Vector Store**: ChromaDB with persistent storage
- **Embeddings**: Ollama embeddings using llama3.2
- **Document Store**: Local file store for raw documents
- **Retrieval**: MultiVectorRetriever for sophisticated retrieval logic

### LLM Pipeline

The RAG chain follows this sequence:

```python
query → retrieve_documents → split_images_texts → create_llm_input → llm → parse_output
```

### Retry Mechanism

All external calls (LLM inference, web scraping) use a retry decorator:

- Default: 3 attempts
- Logs each failure
- Raises exception after all attempts fail

## Development Tools

### Code Quality

The project includes tools for maintaining code quality:

```bash
# Run linter
./lint.sh

# Format code with Black
black .

# Type checking with MyPy
mypy .

# Linting with Ruff
ruff check .
```

### Dependencies

- **LangChain**: RAG orchestration and document processing
- **ChromaDB**: Vector database
- **Ollama**: Local LLM inference
- **BeautifulSoup4**: Web scraping
- **Streamlit**: Web interface
- **DeepEval**: RAG evaluation framework
- **Black/Ruff/MyPy**: Code quality tools

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Error

**Error**: `Connection refused to localhost:11434`

**Solution**:
```bash
# Check if Ollama is running
ollama serve

# Or restart Ollama service
```

#### 2. Model Not Found

**Error**: `Model 'llava:13b' not found`

**Solution**:
```bash
# Pull the missing model
ollama pull llava:13b
```

#### 3. Out of Memory

**Error**: `CUDA out of memory` or system freezes

**Solution**:
- Use smaller models (e.g., `llava:7b` instead of `llava:13b`)
- Reduce chunk size in `config.py`
- Close other applications to free memory

#### 4. Scraping Fails

**Error**: `Unable to get response from URL`

**Solution**:
- Check your internet connection
- Verify The Batch website is accessible
- The date must be a Wednesday when The Batch is published
- Try a different date range

#### 5. Empty Results

**Problem**: RAG returns "I don't know"

**Solution**:
- Ensure database is populated: `python populate_db.py`
- Check `data/chroma_db` and `data/doc_store` directories exist and contain data
- Try repopulating with more issues

#### 6. Slow Performance

**Problem**: Queries take too long

**Solution**:
- Use smaller/faster models
- Reduce the number of retrieved documents
- Ensure you're using the cached UI (`@st.cache_resource`)

### Logging

The system uses comprehensive logging. Check logs for debugging:

```python
from logs import logger

# Logs are output to console by default
# Adjust logging level in logs.py if needed
```

## References

- **The Batch** by DeepLearning.AI for providing excellent AI content
- **Ollama** for local LLM inference
- **LangChain** for RAG orchestration framework
- **ChromaDB** for vector storage
