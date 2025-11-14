# LangChain Examples Collection

A collection of LangChain implementations demonstrating various use cases, from simple prompt engineering to AI agents and RAG (Retrieval Augmented Generation) systems.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Features & Components](#features--components)
  - [1. Simple Prompts](#1-simple-prompts)
  - [2. Chat Model with History](#2-chat-model-with-history)
  - [3. Local Vector Storage (RAG)](#3-local-vector-storage-rag)
  - [4. Manual ReAct Agent](#4-manual-react-agent)
  - [5. LinkedIn Profile Agent](#5-linkedin-profile-agent)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Development](#development)

---

## Overview

This repository contains multiple LangChain implementations showcasing different patterns and use cases:

- **Prompt Engineering**: Simple examples with Ollama and OpenAI
- **Conversational AI**: Chat models with conversation history management
- **RAG Systems**: Document-based question answering with vector storage
- **AI Agents**: Both manual and automated ReAct agent implementations
- **Web Scraping & Integration**: LinkedIn profile analysis with AI agents

## Project Structure

```
langchain/
├── README.md
├── ollama_simple_prompt.py         # Root-level Ollama prompt example
├── openai_simple_prompt.py         # Root-level OpenAI prompt example
│
├── prompt/                          # Simple prompt examples
│   ├── ollama_simple_prompt.py     # Ollama-based prompting
│   └── openai_simple_prompt.py     # OpenAI-based prompting
│
├── history_model/                   # Chat model with history
│   ├── langchain_model.py          # Main model class with chat capabilities
│   ├── history.py                  # Chat history management
│   └── decorators.py               # Retry logic and error handling
│
├── local-vector-storage/            # RAG implementation
│   └── main.py                     # Vector store and document retrieval
│
├── manual_agent/                    # Manual agent implementation
│   └── manual_react_agent.py      # ReAct agent with custom control flow
│
└── simple_agent/                    # LinkedIn analysis agent
    ├── ice_breaker.py              # Main orchestration script
    ├── linkedin_agent.py           # Agent for LinkedIn URL lookup
    ├── linkedin_scraper.py         # LinkedIn profile scraper
    ├── output_parser.py            # Pydantic-based output parsing
    └── web_search_tool.py          # Tavily search integration
```

## Prerequisites

- Python 3.9+
- Ollama installed locally (for Ollama examples)
- OpenAI API key (for OpenAI examples)
- Tavily API key (for web search features)

### Install Ollama

```bash
# Install Ollama from https://ollama.ai
# Then pull the model:
ollama pull llama3.1
```

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd langchain
```

2. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Tavily (web search)
export TAVILY_API_KEY="your-tavily-api-key"
```

## Dependencies

### Core Dependencies

```txt
langchain==0.3.7
langchain-ollama==0.2.0
langchain-openai==0.2.0
langchain-community==0.3.5
langchainhub==0.1.21
pypdf==5.1.0
faiss-cpu==1.9.0
pydantic==2.0+
tenacity>=8.0.0
```

### Development Dependencies

```txt
black==24.10.0
```

Install all dependencies:
```bash
pip install langchain==0.3.7 langchain-ollama==0.2.0 langchain-community==0.3.5 \
    langchainhub==0.1.21 pypdf==5.1.0 faiss-cpu==1.9.0 langchain-openai \
    pydantic tenacity black==24.10.0
```

---

## Features & Components

### 1. Simple Prompts

**Location**: `prompt/` and root directory

Basic examples demonstrating prompt templates and LLM chains with both Ollama and OpenAI.

**Files**:
- `ollama_simple_prompt.py` / `prompt/ollama_simple_prompt.py`
- `openai_simple_prompt.py` / `prompt/openai_simple_prompt.py`

**Features**:
- Simple prompt templates with variables
- Chain composition using LCEL (LangChain Expression Language)
- Information summarization and fact extraction

**Usage**:
```bash
# Using Ollama
python prompt/ollama_simple_prompt.py

# Using OpenAI
python prompt/openai_simple_prompt.py
```

**Example Code**:
```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

summary_template = """
given the information {information} about a person I want you to create:
    1. short summary
    2. two interesting facts about them
"""

prompt_template = PromptTemplate(
    input_variables=["information"],
    template=summary_template
)
llm = ChatOllama(model="llama3.1")
chain = prompt_template | llm

result = chain.invoke(input={"information": "..."})
```

---

### 2. Chat Model with History

**Location**: `history_model/`

A robust chat model implementation with conversation history management, retry logic, and logging capabilities.

**Components**:

#### `langchain_model.py` - Main Model Class
- Configurable chat model with system messages
- Automatic conversation history management
- Built-in retry logic with error handling
- Conversation logging to JSONL format

#### `history.py` - Chat History Management
- Message history tracking (System, Human, AI messages)
- Image message support (base64 encoded)
- Conversation restart functionality
- JSON export and logging

#### `decorators.py` - Utility Decorators
- Retry decorator with configurable attempts
- Custom error messages and callbacks
- Built on `tenacity` library

**Features**:
- ✅ Persistent conversation history
- ✅ System message support
- ✅ Automatic retry on failures
- ✅ Conversation logging (JSONL format)
- ✅ Image message support
- ✅ Restart/reset functionality

**Usage**:
```python
from history_model.langchain_model import Model

# Initialize model
model = Model(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0.7,
    system_message="You are a helpful assistant."
)

# Chat with the model
response = model.chat("What is the capital of France?")
print(response)

# Continue conversation (history is maintained)
response = model.chat("What's the population?")

# Restart conversation
model.restart_chat()

# Log conversation to file
model.log(execution_id="session_001", file_name="conversations.jsonl")
```

**Advanced Features**:
```python
# Add image to conversation
chat_history.add_user_image(image_b64="base64_string", mime="image/png")

# Access raw messages
messages = chat_history.get_messages()

# Remove specific message
chat_history.remove_message(idx=2)
```

---

### 3. Local Vector Storage (RAG)

**Location**: `local-vector-storage/main.py`

Implements Retrieval Augmented Generation (RAG) using FAISS vector store and PDF documents.

**Features**:
- PDF document loading and processing
- Document chunking with overlap
- Vector embeddings using Ollama
- FAISS vector store for efficient similarity search
- Question answering over documents
- Pre-built prompts from LangChain Hub

**How it Works**:
1. Loads PDF documents using `PyPDFLoader`
2. Splits documents into chunks (1000 chars with 30 char overlap)
3. Creates embeddings using Ollama's llama3.1
4. Stores embeddings in FAISS vector store
5. Retrieves relevant documents for queries
6. Generates answers using retrieved context

**Usage**:
```bash
# Make sure you have a PDF file named 'react.pdf' in the directory
python local-vector-storage/main.py
```

**Example Workflow**:
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

# Load and process documents
loader = PyPDFLoader(file_path="react.pdf")
documents = loader.load()

# Create vector store (first time)
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = FAISS.from_documents(documents, embeddings)
vectorstore.save_local("vectorstore")

# Load existing vector store
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# Query the documents
retrieval_chain = create_retrieval_chain(
    vectorstore.as_retriever(),
    combine_docs_chain
)
result = retrieval_chain.invoke({
    "input": "Give me the gist of ReAct in 3 sentences"
})
```

**Configuration**:
- `chunk_size`: 1000 characters
- `chunk_overlap`: 30 characters
- `separator`: `\n`
- Model: llama3.1 (via Ollama)

---

### 4. Manual ReAct Agent

**Location**: `manual_agent/manual_react_agent.py`

A manually implemented ReAct (Reasoning + Acting) agent with custom control flow and tool execution.

**Features**:
- Custom ReAct agent implementation
- Manual control over agent loop
- Tool integration (custom tools)
- Detailed logging with callback handlers
- Scratchpad for intermediate steps
- Stop sequences for controlled generation

**ReAct Pattern**:
```
Question: [User's question]
Thought: [Agent's reasoning]
Action: [Tool to use]
Action Input: [Tool input]
Observation: [Tool output]
... (repeat until answer is found)
Thought: I now know the final answer
Final Answer: [The answer]
```

**Example Tool**:
```python
@tool
def get_text_length(text: str) -> int:
    """Returns length of a text by characters"""
    return len(text.strip())
```

**Usage**:
```bash
python manual_agent/manual_react_agent.py
```

**How it Works**:
1. Agent receives a question
2. Reasons about what to do (Thought)
3. Decides on an action (tool to use)
4. Executes the tool with input
5. Observes the result
6. Repeats until final answer is reached

**Custom Callback Handler**:
- Logs prompts sent to LLM
- Logs LLM responses
- Useful for debugging and monitoring

---

### 5. LinkedIn Profile Agent

**Location**: `simple_agent/`

An intelligent agent system that finds LinkedIn profiles, scrapes them, and generates summaries with interesting facts.

**Components**:

#### `ice_breaker.py` - Main Orchestrator
Coordinates the entire process:
1. Find LinkedIn URL using agent
2. Scrape LinkedIn profile
3. Generate structured summary

#### `linkedin_agent.py` - URL Lookup Agent
- Uses ReAct agent pattern
- Searches web for LinkedIn profiles
- Extracts LinkedIn URL from search results
- Built-in error handling

#### `linkedin_scraper.py` - Profile Scraper
- Scrapes LinkedIn profile data
- Data cleaning and filtering
- Mock mode for testing

#### `output_parser.py` - Structured Output
- Pydantic models for type safety
- Structured output parsing
- Automatic validation

#### `web_search_tool.py` - Search Integration
- Tavily search integration
- URL extraction with regex
- Mock mode for development

**Features**:
- ✅ Automated LinkedIn profile discovery
- ✅ Profile scraping and data extraction
- ✅ AI-powered summary generation
- ✅ Structured output with Pydantic
- ✅ Web search integration
- ✅ Error handling and validation

**Usage**:
```bash
# Run the complete pipeline
python simple_agent/ice_breaker.py

# Or run components individually
python simple_agent/linkedin_agent.py
python simple_agent/linkedin_scraper.py
```

**Example Output**:
```python
{
    "summary": "Experienced software engineer...",
    "facts": [
        "Published author with 5 technical books",
        "Speaker at major tech conferences"
    ]
}
```

**Agent Workflow**:
```
User Input: "Eden Marco Udemy"
    ↓
LinkedIn Agent (searches for profile)
    ↓
URL Extractor (finds LinkedIn URL)
    ↓
LinkedIn Scraper (gets profile data)
    ↓
LLM Chain (generates summary)
    ↓
Output Parser (structures result)
    ↓
Final Output: { summary, facts }
```

**Configuration**:
```python
# Enable real API calls (requires API keys)
linkedin_scraper.scrape_linkedin_profile(url, mock=False)
web_search_tool.get_profile_url_tavily(name, mock=False)
```

---

## Configuration

### Environment Variables

Create a `.env` file or export these variables:

```bash
# OpenAI Configuration
export OPENAI_API_KEY="sk-..."

# Tavily Search API (for web search)
export TAVILY_API_KEY="tvly-..."

# Ollama Configuration (if using remote instance)
export OLLAMA_BASE_URL="http://localhost:11434"
```

### Model Configuration

**Ollama Models**:
```bash
# Pull models
ollama pull llama3.1
ollama pull mistral
ollama pull codellama

# List available models
ollama list

# Run Ollama server (if not running)
ollama serve
```

**OpenAI Models**:
- `gpt-3.5-turbo` - Fast and cost-effective
- `gpt-4` - More capable, higher cost
- `gpt-4-turbo` - Latest capabilities

---

## Usage Examples

### Example 1: Simple Question Answering

```python
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

template = "Answer this question: {question}"
prompt = PromptTemplate(input_variables=["question"], template=template)
llm = ChatOllama(model="llama3.1")
chain = prompt | llm

result = chain.invoke({"question": "What is LangChain?"})
print(result.content)
```

### Example 2: Chat with History

```python
from history_model.langchain_model import Model

model = Model(
    base_url="http://localhost:11434",
    model="llama3.1",
    temperature=0.7,
    system_message="You are a Python expert."
)

# Multi-turn conversation
print(model.chat("What is a decorator in Python?"))
print(model.chat("Can you show me an example?"))
print(model.chat("What are common use cases?"))

# Save conversation
model.log(execution_id="python_tutorial")
```

### Example 3: Document Q&A with RAG

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load documents
loader = PyPDFLoader("your_document.pdf")
docs = loader.load()

# Create vector store
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = FAISS.from_documents(docs, embeddings)

# Create QA chain
llm = OllamaLLM(model="llama3.1")
prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(
    vectorstore.as_retriever(),
    combine_docs_chain
)

# Ask questions
result = retrieval_chain.invoke({"input": "What is the main topic?"})
print(result["answer"])
```

### Example 4: LinkedIn Profile Analysis

```python
from simple_agent.linkedin_agent import linkedin_lookup_agent
from simple_agent.linkedin_scraper import scrape_linkedin_profile
from simple_agent.ice_breaker import chain

# Find and analyze profile
linkedin_url = linkedin_lookup_agent(name="John Doe")
profile_data = scrape_linkedin_profile(linkedin_url)
summary = chain.invoke({"information": profile_data})

print(f"Summary: {summary.summary}")
print(f"Facts: {summary.facts}")
```

### Example 5: Custom Agent with Tools

```python
from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain import hub

@tool
def calculator(expression: str) -> float:
    """Evaluates a mathematical expression"""
    return eval(expression)

@tool
def word_counter(text: str) -> int:
    """Counts words in text"""
    return len(text.split())

# Create agent
llm = ChatOllama(model="llama3.1")
tools = [calculator, word_counter]
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Use agent
result = agent_executor.invoke({
    "input": "Calculate 15 * 23 and count words in 'Hello world from LangChain'"
})
print(result["output"])
```

---

## Development

### Code Formatting

This project uses Black for code formatting:

```bash
# Format all files
black .

# Format specific directory
black history_model/

# Check formatting without changing files
black --check .
```

### Running Examples

```bash
# Simple prompts
python prompt/ollama_simple_prompt.py
python prompt/openai_simple_prompt.py

# Chat with history
python -c "from history_model.langchain_model import Model; \
    m = Model('http://localhost:11434', 'llama3.1', 0.7); \
    print(m.chat('Hello!'))"

# Vector storage RAG
python local-vector-storage/main.py

# Manual agent
python manual_agent/manual_react_agent.py

# LinkedIn agent
python simple_agent/ice_breaker.py
```

### Project Best Practices

1. **Use virtual environments**: Always work within a virtual environment
2. **Environment variables**: Store API keys in `.env` file (never commit it!)
3. **Mock mode**: Use mock mode during development to avoid API costs
4. **Logging**: Enable verbose mode for debugging agents
5. **Type hints**: Use type hints for better code quality
6. **Error handling**: Implement retry logic for production systems

### Common Issues & Solutions

**Issue**: `ModuleNotFoundError: No module named 'langchain'`
```bash
pip install langchain langchain-ollama langchain-community
```

**Issue**: Ollama connection refused
```bash
# Start Ollama server
ollama serve

# Check if running
curl http://localhost:11434
```

**Issue**: OpenAI API key not found
```bash
export OPENAI_API_KEY="your-key-here"
# Or add to .env file
```

**Issue**: FAISS vector store loading error
```python
# Use allow_dangerous_deserialization=True (only with trusted data)
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
```

---

## Advanced Topics

### Custom Prompts

Create sophisticated prompts with multiple variables:

```python
from langchain_core.prompts import PromptTemplate

template = """
Context: {context}
Question: {question}
Language: {language}

Provide a {style} answer in {language}.
"""

prompt = PromptTemplate(
    input_variables=["context", "question", "language", "style"],
    template=template
)
```

### Memory Types

LangChain supports various memory types:

- **ConversationBufferMemory**: Stores all messages
- **ConversationBufferWindowMemory**: Stores last N messages
- **ConversationSummaryMemory**: Stores a summary of the conversation
- **ConversationTokenBufferMemory**: Stores based on token count

### Callbacks

Monitor agent execution with callbacks:

```python
from langchain.callbacks.base import BaseCallbackHandler

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"Prompt: {prompts[0]}")

    def on_llm_end(self, response, **kwargs):
        print(f"Response: {response}")

llm = ChatOllama(model="llama3.1", callbacks=[CustomHandler()])
```

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---
