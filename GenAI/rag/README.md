# RAG

## Dependencies

* ollama - https://ollama.com/download
* python3 - https://www.python.org/downloads/

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

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:latest
ollama pull llama3.2:latest
