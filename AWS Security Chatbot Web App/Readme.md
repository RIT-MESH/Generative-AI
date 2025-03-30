# AWS Security Chatbot Project Documentation

## Project Overview
This project demonstrates how to build a **Retrieval-Augmented Generation (RAG) Chatbot** using **Amazon Bedrock**, with a focus on AWS Security. The chatbot is accessible via a **FastAPI-based API** and a **custom web interface**, allowing users to query uploaded AWS Security documents.


![Screenshot 2025-03-30 173957](https://github.com/user-attachments/assets/f420f352-06ca-4cb4-8853-23f844d4b2c9)


![Screenshot 2025-03-30 174955](https://github.com/user-attachments/assets/ec34f7f4-c3f4-4404-9cea-4c082a28c6f7)


![Screenshot 2025-03-30 175225](https://github.com/user-attachments/assets/370a60d8-b370-45b4-acb2-5eddcde05165)


## Features
- Uses Amazon Bedrock Knowledge Base for document retrieval.
- Meta LLaMA 3 70B Instruct model for response generation.
- RESTful API with FastAPI.
- Interactive frontend built with HTML, CSS, and JavaScript.
- Chat interface with Enter key handling and Clear Chat button.

---

## Tools and Services Used
- **Amazon S3**: Storage for PDF/Docs.
- **Amazon Bedrock**: Embedding and LLM access.
- **OpenSearch Serverless**: Vector store.
- **FastAPI**: Backend API.
- **Boto3**: AWS SDK for Python.
- **Dotenv**: Manage secrets with .env file.
- **HTML/CSS/JavaScript**: Frontend for chat UI.

---

## Steps to Build the Project

### Step 1: Setup AWS Resources
1. **Create S3 Bucket** in `us-east-2` region and upload AWS Security PDF documents.
2. **Enable Bedrock Models**:
   - Titan Text Embeddings v2
   - Meta LLaMA 3 70B Instruct
3. **Create a Knowledge Base** in Amazon Bedrock using your S3 documents.
4. **Select Vector Store**: Choose OpenSearch Serverless with quick create.
5. **Sync** the Knowledge Base after it's created.

### Step 2: Setup Local CLI and Credentials
1. Install **AWS CLI**.
2. Create and download **IAM access keys**.
3. Run `aws configure` with your credentials and set region as `us-east-2`.
4. Test using CLI:
```bash
aws bedrock-agent-runtime retrieve-and-generate \
  --input '{"text": "What is AWS IAM?"}' \
  --retrieve-and-generate-configuration '{
    "knowledgeBaseConfiguration": {
      "knowledgeBaseId": "<ID>",
      "modelArn": "<ARN>"
    },
    "type": "KNOWLEDGE_BASE"
}'
```

### Step 3: Build API using FastAPI
1. Clone the GitHub repo or create your own project folder.
2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Create `.env` file with:
```env
AWS_REGION=us-east-2
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
KNOWLEDGE_BASE_ID=...
MODEL_ARN=...
```
5. Start API:
```bash
python -m uvicorn main:app --reload
```
6. Visit `http://127.0.0.1:8000/bedrock/query?text=your+question`

### Step 4: Build Web Frontend
1. Create `index.html` file with chat interface.
2. Handle user input and response display using JavaScript.
3. Support Enter key and Clear Chat functionality.
4. Open `index.html` in browser and chat away!

---

## Problems Faced & Solutions

### 1. **CORS Issue Between Frontend and Backend**
- **Problem**: Frontend couldn’t connect to FastAPI backend due to CORS error.
- **Fix**: Added `CORSMiddleware` in `main.py` with `allow_origins=["*"]`.

### 2. **Enter Key Not Triggering Message Send**
- **Problem**: Only the Send button worked, Enter key did nothing.
- **Fix**: Added JavaScript event listener for `keypress` with `e.key === "Enter"`.

### 3. **Response Not Readable (Too Long)**
- **Problem**: Long text responses showed as a single line.
- **Fix**: Used `white-space: pre-wrap` in CSS to wrap long bot messages.

### 4. **Invalid Credentials / No Output**
- **Problem**: Bedrock API calls returned access errors.
- **Fix**: Used `.env` file with credentials and verified using Boto3 session in code.

### 5. **Message Overlapping or Not Scrolling**
- **Fix**: Set `scrollTop = scrollHeight` in JS after message update.

---

## Final Result
✅ Chatbot UI
✅ Working backend API
✅ Integrated Bedrock RAG with AWS Security knowledge base
✅ Clean and secure credential handling

---

## Future Improvements
- Add Markdown support to format responses
- Deploy on EC2 or use API Gateway + Lambda for public access
- Add chat history and user sessions

---




