# Multimodal RAG

This is a multimodal rag project for *The Batch* that takes information from the text and the images found in the articles.

## Dependencies

* ollama - https://ollama.com/download
* python3.12.10 - https://www.python.org/downloads/release/python-31210/

### Python

* langchain==0.3.19
* langchain-chroma==0.2.2
* langchain-community==0.3.18
* langchain-ollama==0.3.3
* langchain-text-splitters==0.3.6
* beautifulsoup4==4.13.3
* requests==2.32.3
* ruff==0.11.10
* black==25.1.0
* mypy==1.15.0
* deepeval==2.9.4
* streamlit==1.42.2

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2
ollama pull llava:13b
ollama pull gemma3
ollama pull deepseek-r1:1.5b
```

## Execution

To extract data from The Batch

```
python populate_db.py -h  # for execution instructions
python populate_db.py <start_date> <num_issues>
```

To execute Rag in headless mode

```
python rag.py "<query>"
```

To execute Rag with UI

```
streamlit run ui.py
```

## Test

```
deepeval set-ollama deepseek-r1:1.5b
python test_rag.py
```
