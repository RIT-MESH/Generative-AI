## 🚀 LangChain + LangSmith + LangServe Quickstart Guide (with AWS DevOps Humor & Explanations)

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

#### A. Custom Prompt Template (with Joke + Explanation)

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "You're a cloud security expert with a sense of humor. Tell a short joke about {topic}, then explain why it's funny."
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

**Example Output:**
> **Joke:** Why did the developer get locked out of production?  
> Because they finally fixed that overly permissive IAM policy!  
>  
> **Explanation:** IAM policies define what users can and can't do. Overly permissive ones are risky, but once tightened, users may lose access they relied on — hence, “locked out.”

---

### 📊 Step 3: Trace with LangSmith

#### A. Set environment variables:
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

---

### 🌐 Step 4: Serve with LangServe

#### A. `main.py` for LangServe:
```python
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template(
    "You're a cloud security expert with a sense of humor. Tell a short joke about {topic}, then explain why it's funny."
)
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

app = FastAPI()
add_routes(app, prompt | llm | parser, path="/joke")
```

#### B. Run your server:
```bash
uvicorn main:app --reload
```

#### C. Test the endpoint:
```bash
curl -X POST http://localhost:8000/joke/invoke -H "Content-Type: application/json" -d "{\"input\": {\"topic\": \"EC2 security groups\"}}"
```

---

### ✅ Summary

You now have:
- A LangChain app using a **humorous cloud security expert**
- **LangSmith tracing** enabled
- A **LangServe API** you can call from anywhere

---
