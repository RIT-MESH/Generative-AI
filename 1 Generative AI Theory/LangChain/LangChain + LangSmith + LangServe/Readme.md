## 🚀 LangChain + LangSmith + LangServe Quickstart Guide (with AWS DevOps Humor)

### ✅ Prerequisites

- Python 3.10+
- Conda or virtualenv
- VS Code or any code editor
- OpenAI API key (or any model provider key)

---

### 📁 Step 1: Environment Setup

#### A. Create and activate Conda env:
```bash
conda create -p venv python=3.10 -y
conda activate ./venv
```

#### B. Install dependencies:
```bash
pip install langchain langsmith langserve openai
```

---

### 🔧 Step 2: Set Up LangChain Components

#### A. Custom Prompt Template for AWS DevOps / Security

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "You're a cloud security expert with a sense of humor. Tell a security joke about {topic}."
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()
```

#### B. Chain it all together

```python
chain = prompt | llm | parser
```

#### C. Run the chain
```python
response = chain.invoke({"topic": "IAM policies"})
print(response)
```

---

### 📊 Step 3: Trace with LangSmith

#### A. Set environment variables (you can add these to `.env` or your terminal):
```bash
export LANGCHAIN_API_KEY=your_langsmith_api_key
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=quickstart
```

> On Windows CMD:
```cmd
set LANGCHAIN_API_KEY=your_langsmith_api_key
set LANGCHAIN_TRACING_V2=true
set LANGCHAIN_PROJECT=quickstart
```

Now all runs using `chain.invoke()` will be traced and visible in your LangSmith dashboard!

---

### 🌐 Step 4: Serve with LangServe

#### A. Create a `main.py`:
```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "You're a cloud security expert with a sense of humor. Tell a security joke about {topic}."
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

app = FastAPI()
add_routes(app, prompt | llm | parser, path="/joke")
```

#### B. Start the server:
```bash
uvicorn main:app --reload
```

#### C. Access via browser or curl:
```bash
http://localhost:8000/joke/invoke
```

Example POST body:
```json
{"input": {"topic": "AWS security groups"}}
```

---

### ✅ Summary

You now have:
- A **LangChain** app with a custom cloud security humor prompt
- Live **LangSmith** tracing
- A running **LangServe** API locally

---
