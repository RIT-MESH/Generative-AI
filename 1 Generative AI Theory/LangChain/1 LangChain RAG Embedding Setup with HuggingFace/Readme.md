# Project Documentation: LangChain RAG Embedding Setup with HuggingFace

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot using LangChain with HuggingFace embeddings. The workflow involves document ingestion, vector embedding, and chatbot query handling through a local or hosted LLM.

## Tools & Libraries
The setup involves the following key packages:

- `langchain`
- `langchain-community`
- `sentence-transformers`
- `langchain-huggingface`
- `transformers`
- `torch`
- `faiss-cpu` or `chromadb` for vector store

Refer to the included `requirements.txt` for the complete environment dependencies.

## Key Errors & Solutions

### 1. **Deprecation Warning for `HuggingFaceEmbeddings`**

#### Problem
```
LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2
```

#### Solution
Install and import from the updated `langchain-huggingface` package:

```bash
pip install -U langchain-huggingface
```

**Correct Import:**
```python
from langchain_huggingface import HuggingFaceEmbeddings
```

### 2. **Environment Variable Setup (GROQ_API_KEY / HF_TOKEN)**

#### Problem
Environment variable not being recognized:
```
'GROQ_API_KEY' is not recognized as an internal or external command
```

#### Solution
Instead of trying to assign variables in the terminal directly, use a `.env` file and `python-dotenv`:

**.env**
```
GROQ_API_KEY=gsk_abc123...
HF_TOKEN=your_hf_token_here
```

**Python code:**
```python
from dotenv import load_dotenv
load_dotenv()
```

### 3. **`NameError: name 'init_empty_weights' is not defined`**

#### Problem
Using `HuggingFaceEmbeddings` with older version of `transformers`/`torch` caused this:
```
NameError: name 'init_empty_weights' is not defined
```

#### Solution
Upgrade packages to supported versions:
```bash
pip install --upgrade torch transformers sentence-transformers
```

**Verified Versions:**
- torch >= 2.0.0
- transformers >= 4.37.2
- sentence-transformers >= 2.5.1

Also, clearing and reinitializing model cache may help:
```python
from transformers.utils import move_cache
move_cache()
```

### 4. **Jupyter Notebook Setup (for VS Code users)**

If using VS Code and notebook kernels were broken or widgets not rendering:
```bash
pip install -U jupyter ipywidgets
```

### 5. **Transformers Cache Migration Warning**
```
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache.
```

#### Solution
This is not an error, just a one-time migration log. Let it complete or use:
```python
from transformers.utils import move_cache
move_cache()
```

## Summary of Fixes
| Issue | Resolution |
|-------|------------|
| Deprecated `HuggingFaceEmbeddings` | Use `langchain-huggingface` |
| API Keys not recognized | Use `.env` and `load_dotenv()` |
| `init_empty_weights` error | Upgrade `torch`/`transformers`/`sentence-transformers` |
| Widget / notebook errors in VS Code | Install `jupyter`, `ipywidgets` |
| Cache warning | Use `transformers.utils.move_cache()` or ignore |

---
This document should serve as a quick reference for resolving environment and package compatibility issues encountered while setting up LangChain RAG pipelines with HuggingFace models.

