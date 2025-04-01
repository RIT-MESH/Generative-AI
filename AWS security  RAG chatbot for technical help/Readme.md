# RAG Chatbot with Amazon Bedrock - Step-by-Step Documentation

## What is RAG (Retrieval-Augmented Generation)?
RAG is a technique that combines a knowledge retrieval system with a generative AI model. This enables AI to generate more accurate and personalized answers by referencing your own documents.

In this project, you will:
- Upload documents to Amazon S3.
- Create a Knowledge Base in Amazon Bedrock.
- Use vector search with OpenSearch Serverless.
- Enable AI models in Bedrock.
- Use AWS CloudShell and CLI to interact with your chatbot.

---

## Step 1: Store Your Documents in Amazon S3

### 1.1 Create an S3 Bucket
1. Go to the AWS Console and switch to region **Ohio (us-east-2)**.
2. In the AWS search bar, type **S3** and click on **S3**.
3. Click **Create bucket**.
4. Bucket name: `S3-AWS-Security-rag-docs`
5. Leave **ACLs disabled** and **Block all public access** checked.
6. Leave encryption settings as default.
7. Click **Create bucket**.

### 1.2 Upload Documents
1. Click into your newly created bucket.
2. Click **Upload**.
3. Click **Add files**, select your 10 PDF documents.
4. Click **Upload**.

---

## Step 2: Create a Knowledge Base in Bedrock

### 2.1 Navigate to Bedrock
1. In the AWS Console, type **Bedrock** in the search bar and click **Amazon Bedrock**.
2. In the left sidebar, under **Builder Tools**, click **Knowledge Bases**.
3. Click **Create** > **Knowledge Base with vector store**.

### 2.2 Configure Knowledge Base
1. Name: `AWS-security-rag-documentation`
2. Description: `This Knowledge Base stores all documentation at NextWork.`
3. IAM Role: Select **Create and use a new service role**.

### 2.3 Choose Data Source
1. Select **Amazon S3** as the data source.
2. Data Source Name: `s3-bucket-AWS-Security-rag-documentation`
3. Location: **This AWS account**
4. Click **Browse S3**, select your bucket.
5. Click **Choose**.
6. Leave default parsing and chunking strategy.

### 2.4 Select Embedding Model
1. Click **Select model**.
2. Choose **Titan Text Embeddings v2**.
3. Click **Apply**.

### 2.5 Vector Store
1. Choose **Quick create a new vector store**.
2. Select **Amazon OpenSearch Serverless**.
3. Click **Next**.

### 2.6 Review and Create
1. Confirm configuration.
2. Click **Create Knowledge Base**.

---

## Step 3: Sync Knowledge Base
1. In the Knowledge Base page, go to **Data Sources**.
2. Check the box next to your S3 source.
3. Click **Sync**.

This processes and indexes your documents.

---

## Step 4: Enable AI Models

### 4.1 Navigate to Model Access
1. In Bedrock, scroll down to **Bedrock Configurations > Model access**.
2. Click **Enable specific models**.
3. Select:
   - **Titan Text Embeddings V2**
   - **Meta Llama 3.3 70B Instruct**
4. Click **Next**, agree to the terms, and click **Submit**.

---

## Step 5: Test in AWS CloudShell

### 5.1 Open CloudShell
1. Click the **CloudShell icon** on the top-right of the AWS Console.
2. Wait for the terminal to initialize.

### 5.2 Verify CLI
```bash
aws --version
```

### 5.3 Set Up a Bash Function to Ask Questions
```bash
ask_bedrock() {
  QUESTION="$1"
  aws bedrock-agent-runtime retrieve-and-generate \
    --region us-east-2 \
    --input "{\"text\": \"$QUESTION\"}" \
    --retrieve-and-generate-configuration '{
      "knowledgeBaseConfiguration": {
        "knowledgeBaseId": "REPLACE_WITH_YOUR_KB_ID",
        "modelArn": "arn:aws:bedrock:us-east-2::foundation-model/meta.llama3-3-70b-instruct-v1:0"
      },
      "type": "KNOWLEDGE_BASE"
    }' \
    --query 'output.text' \
    --output text
}
```

Replace `REPLACE_WITH_YOUR_KB_ID` with your actual Knowledge Base ID.

### 5.4 Ask a Question
```bash
ask_bedrock "What is the AWS Well-Architected Framework?"
```

---

## Recap: Services Used
- **Amazon S3** – to store your documents.
- **Amazon Bedrock** – to manage your Knowledge Base and AI models.
- **Titan Text Embeddings v2** – to create semantic vectors.
- **Meta Llama 3.3 70B Instruct** – to generate human-like responses.
- **AWS CloudShell** – to run commands in the terminal.

---

## Cleanup (Important!)
To avoid charges, delete these resources after use:
1. Delete the **Knowledge Base** in Bedrock.
2. Delete the **S3 bucket**.
3. Delete the **OpenSearch Vector Store**.

---

You're now ready to build, deploy, and scale a personalized RAG chatbot with Amazon Bedrock via CLI! 🚀

