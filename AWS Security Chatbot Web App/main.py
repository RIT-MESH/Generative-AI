from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ✅ Check if credentials are loading correctly
session = boto3.session.Session()
creds = session.get_credentials().get_frozen_credentials()
print("Access Key:", creds.access_key)
print("Secret Key:", creds.secret_key[:4] + "..." + creds.secret_key[-4:])  # hide part for safety


# Retrieve AWS configuration from environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
KNOWLEDGE_BASE_ID = os.getenv("KNOWLEDGE_BASE_ID")
MODEL_ARN = os.getenv("MODEL_ARN")

app = FastAPI()

# ✅ Enable CORS so HTML frontend can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to ["http://localhost:8080"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Boto3 client for Bedrock Agent Runtime
def get_bedrock_client():
    return boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

@app.get("/")
async def root():
    return {"message": "Welcome to your RAG chatbot API!"}

@app.get("/bedrock/query")
async def query_bedrock(text: str = Query(..., description="Input text for the model")):
    client = get_bedrock_client()
    try:
        response = client.retrieve_and_generate(
            input={"text": text},
            retrieveAndGenerateConfiguration={
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn": MODEL_ARN
                },
                "type": "KNOWLEDGE_BASE"
            }
        )
        return {"response": response["output"]["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
