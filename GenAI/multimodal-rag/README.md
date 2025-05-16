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
* ruff==0.11.10
* black==25.1.0
* mypy==1.15.0
* requests==2.32.3

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1
ollama pull bakllava
ollama pull llama3.2-vision
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
deepeval test run test_rag.py
```
