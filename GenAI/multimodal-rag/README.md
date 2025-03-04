# Multimodal RAG

## Dependencies

* ollama - https://ollama.com/download
* python3 - https://www.python.org/downloads/

### Python

* langchain==0.3.19
* langchain-core==0.3.40
* langchain-text-splitters==0.3.6
* langchain-community==0.3.18
* langchain-text-splitters==0.3.6
* langchain-chroma==0.2.2
* beautifulsoup4==4.13.3
* streamlit==1.42.2
* deepeval==2.4.9

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

```
ollama pull llama3.1
ollama pull bakllava
ollama pull llama3.2-vision
```

## Execution

To extract data from The Batch

```
python populate_db.py
```

To execute Rag in headless mode

```
python rag.py "<query>"
```
