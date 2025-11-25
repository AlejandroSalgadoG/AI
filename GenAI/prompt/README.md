# DSPy Examples

A collection of DSPy framework examples for algorithmic prompt optimization and structured LLM interactions.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Examples](#examples)
  - [1. Basic QA](#1-basic-qa)
  - [2. Chain of Thought (CoT)](#2-chain-of-thought-cot)
  - [3. Typed Predictors](#3-typed-predictors)
  - [4. Retrieval-Augmented Generation (RAG)](#4-retrieval-augmented-generation-rag)
- [Running the Examples](#running-the-examples)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)
- [Resources](#resources)

## Overview

This repository contains practical examples of using [DSPy](https://github.com/stanfordnlp/dspy) (Declarative Self-improving Language Programs), a framework that transforms how we build LLM-based applications. Rather than manually crafting and maintaining prompts, DSPy allows you to define signatures and modules that can be optimized algorithmically.

These examples are inspired by the "Complete DSPy Tutorial" by Neural Breakdown with AVB.

## Motivation

Traditional prompt engineering faces several challenges:

- **Brittle prompts**: Prompts may need complete rewrites when switching LLM providers or versions
- **Module dependencies**: Output quality depends heavily on previous modules working correctly
- **Manual supervision**: Requires constant human intervention to tune and maintain prompts
- **Over-tuning**: Prompts optimized for one LLM may perform poorly on newer models
- **Lack of composability**: Difficult to chain multiple prompting strategies together

**DSPy addresses these issues by:**

- Providing a declarative way to define LLM interactions through signatures
- Enabling algorithmic optimization of prompts
- Supporting modular composition of reasoning patterns
- Automatically adapting to different LLMs
- Offering structured outputs with type safety

## Prerequisites

Before running these examples, ensure you have:

1. **Python 3.12+** installed on your system
2. **Ollama** running locally with the `llama3.1` model
   ```bash
   # Install Ollama from https://ollama.ai
   # Pull the model
   ollama pull llama3.1
   ```
3. **(Optional)** For RAG example: Access to ColBERTv2 retrieval service

## Installation

### Using uv (recommended)

```bash
# Install uv if you haven't already
pip install uv

# Install dependencies
uv sync
```

### Using pip

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install dspy>=3.0.4 pydantic>=2.12.4
```

## Configuration

The project uses a local Ollama instance by default. Configuration is managed in `utils.py`:

```python
def get_lm():
    return dspy.LM(
        "ollama_chat/llama3.1",
        api_base="http://localhost:11434",
        api_key="",
    )
```

To use a different LLM provider (OpenAI, Anthropic, etc.), modify this function accordingly.

## Examples

### 1. Basic QA

**Files**: `basic_qa.py`, `basic_qa_signature.py`

Learn the fundamentals of DSPy with simple question-answering.

#### Basic Predict (`basic_qa.py`)

The simplest form of DSPy interaction using string-based signatures:

```python
predict = dspy.Predict("question -> answer")
prediction = predict(question="What is the capital of France?")
```

**Key features:**
- String-based signature definition
- Automatic system message generation
- Structured input/output format

#### Signature Classes (`basic_qa_signature.py`)

A more structured approach using signature classes:

```python
class QuestionSignature(dspy.Signature):
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="should be between 1 and 5 words")

predict = dspy.Predict(QuestionSignature)
```

**Advantages:**
- Better type safety
- Reusable signatures
- Clear field descriptions that guide the LLM

**Run it:**
```bash
python basic_qa.py
python basic_qa_signature.py
```

### 2. Chain of Thought (CoT)

**Files**: `cot.py`, `cot_module.py`

Implement reasoning patterns for complex questions.

#### Simple Chain of Thought (`cot.py`)

Adds intermediate reasoning steps to improve answer quality:

```python
cot_predict = dspy.ChainOfThought(QuestionSignature)
prediction = cot_predict(question=question)

# Access reasoning and answer
print(prediction.reasoning)  # The thought process
print(prediction.answer)     # The final answer
```

**When to use CoT:**
- Multi-hop reasoning questions
- Questions requiring factual lookup
- Complex analytical tasks

**Example question:**
> "What is the birth state of the winner of the Nobel Prize of Peace in 2009?"

The model will reason through: identify the winner (Barack Obama) → recall his birthplace (Hawaii).

#### Module Composition (`cot_module.py`)

Build reusable modules by composing multiple reasoning steps:

```python
class DoubleCoT(dspy.Module):
    def __init__(self, question: str):
        self.cot1 = dspy.ChainOfThought(QuestionSignature)
        self.cot2 = dspy.ChainOfThought(ThoughtSignature)

    def forward(self, question: str) -> str:
        thought = self.cot1(question=question)
        answer = self.cot2(question=question, thought=thought.answer)
        return dspy.Prediction(thought=thought.answer, answer=answer.answer)
```

**Benefits:**
- Encapsulates complex reasoning pipelines
- Reusable across different questions
- Can be optimized as a unit

**Example question:**
> "What is the second largest city of the birth state of the winner of the Nobel Prize of Peace in 2009?"

This requires two reasoning steps:
1. First CoT: Identify the birth state (Hawaii)
2. Second CoT: Find the second largest city (using the first answer)

**Run it:**
```bash
python cot.py
python cot_module.py
```

### 3. Typed Predictors

**Files**: `typed_predictors.py`, `typed_predictors_list.py`

Get structured outputs using Pydantic models for type safety and validation.

#### Confidence Scores (`typed_predictors.py`)

Return structured data with confidence levels:

```python
class AnswerConfidence(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="The confidence in the answer between 0 and 1")

class QuestionConfidenceSignature(dspy.Signature):
    question: str = dspy.InputField(description="The question to answer")
    answer: AnswerConfidence = dspy.OutputField(description="The answer and confidence")

cot = dspy.ChainOfThought(QuestionConfidenceSignature)
prediction = cot(question=question)
```

**Output:**
```python
{
    "answer": "Barack Obama",
    "confidence": 0.95
}
```

#### List Outputs (`typed_predictors_list.py`)

Generate structured lists of data:

```python
class Answer(BaseModel):
    country: str = Field(description="The country of the answer")
    year: int = Field(description="The year of the answer")

class AnswerListSignature(dspy.Signature):
    """Given the user's question, generate the answer with a JSON readable python list"""
    question: str = dspy.InputField(description="The question to answer")
    answer: list[Answer] = dspy.OutputField(description="The answer to the question")
```

**Example question:**
> "Generate a list of country and the year of FIFA world cup winners from 2002 to 2022"

**Output:**
```python
[
    {"country": "Brazil", "year": 2002},
    {"country": "Italy", "year": 2006},
    {"country": "Spain", "year": 2010},
    # ... etc
]
```

**Use cases:**
- Extracting structured data from text
- Data validation and transformation
- Building type-safe API responses

**Run it:**
```bash
python typed_predictors.py
python typed_predictors_list.py
```

### 4. Retrieval-Augmented Generation (RAG)

**File**: `rag.py`

Combine retrieval with generation for knowledge-grounded answers.

```python
class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        self.retrieve = dspy.Retrieve(k=k)
        self.answer = dspy.ChainOfThought(Answer)

    def forward(self, question: str) -> str:
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)
```

**Components:**
- `dspy.Retrieve`: Fetches top-k relevant passages from a knowledge base
- `dspy.ChainOfThought`: Reasons over the retrieved context to answer the question

**Setup:**

The example uses ColBERTv2 with Wikipedia abstracts:

```python
colbertv2_wiki = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.settings.configure(lm=lm, rm=colbertv2_wiki)
```

**Note:** You can download the Wikipedia abstracts index from:
https://downloads.cs.stanford.edu/nlp/data/colbert/baleen/wiki.abstracts.2017.tar.gz

**Example question:**
> "What is the date of birth of the player who provided the assist for the final goal in the World Cup finals in 2014?"

The system will:
1. Retrieve relevant Wikipedia passages about the 2014 World Cup final
2. Reason over the context to find the answer

**Run it:**
```bash
python rag.py
```

## Running the Examples

Each example can be run independently:

```bash
# Basic QA
python basic_qa.py
python basic_qa_signature.py

# Chain of Thought
python cot.py
python cot_module.py

# Typed Predictors
python typed_predictors.py
python typed_predictors_list.py

# RAG (requires ColBERTv2 setup)
python rag.py
```

## Key Concepts

### Signatures

Signatures define the input-output behavior of LLM modules:

```python
class MySignature(dspy.Signature):
    input_field = dspy.InputField(desc="Description for LLM")
    output_field = dspy.OutputField(desc="Description for LLM")
```

### Modules

Modules encapsulate reusable reasoning patterns:

```python
class MyModule(dspy.Module):
    def __init__(self):
        self.predictor = dspy.ChainOfThought(MySignature)

    def forward(self, input_data):
        return self.predictor(input_field=input_data)
```

### Predictors

Built-in predictors for different reasoning patterns:

- `dspy.Predict`: Basic input-output prediction
- `dspy.ChainOfThought`: Adds reasoning steps before answer
- `dspy.ReAct`: Reasoning and acting in cycles
- `dspy.MultiChainComparison`: Compare multiple reasoning chains

### Structured Outputs

Use Pydantic models for type-safe outputs:

```python
class StructuredOutput(BaseModel):
    field1: str
    field2: int
    field3: list[str]
```

## Project Structure

```
.
├── basic_qa.py              # Basic predict with string signatures
├── basic_qa_signature.py    # Signature-based predict
├── cot.py                   # Simple Chain of Thought
├── cot_module.py            # Composable CoT modules
├── typed_predictors.py      # Structured outputs with confidence
├── typed_predictors_list.py # List-based structured outputs
├── rag.py                   # Retrieval-Augmented Generation
├── utils.py                 # LLM configuration utilities
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```
