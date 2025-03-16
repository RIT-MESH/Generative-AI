# Real-Time & Batch Inference on AWS

## Table of Contents
1. [Batch Processing vs. Real-Time Inference](#batch-processing-vs-real-time-inference)
2. [Deploying Real-Time Models with SageMaker Endpoints](#deploying-real-time-models-with-sagemaker-endpoints)
3. [Asynchronous Inference with SageMaker Asynchronous Mode](#asynchronous-inference-with-sagemaker-asynchronous-mode)
4. [Running ML Inference with AWS Lambda & API Gateway](#running-ml-inference-with-aws-lambda--api-gateway)
5. [Example: Deploying a Fraud Detection API Using SageMaker](#example-deploying-a-fraud-detection-api-using-sagemaker)

---

## 1. Batch Processing vs. Real-Time Inference

### **Batch Processing**
Batch inference is used for processing large volumes of data at scheduled intervals. It is suitable for use cases where latency is not a concern, such as predictive analytics, fraud detection, and recommendation systems.

**Advantages:**
- Cost-efficient as it does not require always-on resources.
- Processes large amounts of data in parallel.
- Suitable for training data augmentation and analytics.

**AWS Services for Batch Inference:**
- **SageMaker Batch Transform**
- **AWS Glue for ML Data Processing**
- **AWS Lambda for Scheduled ML Jobs**

### **Real-Time Inference**
Real-time inference provides immediate predictions, making it ideal for applications such as fraud detection, chatbots, and personalized recommendations.

**Advantages:**
- Low-latency response times.
- Supports event-driven architectures.
- Suitable for dynamic applications like fraud detection and real-time recommendations.

**AWS Services for Real-Time Inference:**
- **SageMaker Endpoints**
- **AWS Lambda & API Gateway**
- **SageMaker Asynchronous Inference**

---

## 2. Deploying Real-Time Models with SageMaker Endpoints

Amazon SageMaker Endpoints provide a scalable and cost-effective way to serve real-time ML predictions.

### **Step-by-Step Process:**
1. **Train and Save the Model in Amazon S3:**
   ```python
   from sagemaker.tensorflow import TensorFlow
   estimator = TensorFlow(entry_point='train.py',
                          role=role,
                          instance_count=1,
                          instance_type='ml.m5.large')
   estimator.fit({'train': 's3://training-data/'})
   ```

2. **Deploy the Model as a SageMaker Endpoint:**
   ```python
   predictor = estimator.deploy(instance_type='ml.m5.large', initial_instance_count=1)
   ```

3. **Invoke the Endpoint for Predictions:**
   ```python
   response = predictor.predict(input_data)
   print(response)
   ```

4. **Monitor Endpoint Performance in CloudWatch:**
   - Track latency, error rates, and throughput using **Amazon CloudWatch Metrics**.

---

## 3. Asynchronous Inference with SageMaker Asynchronous Mode

SageMaker Asynchronous Inference is used when inference requests take a long time to process or need to handle high volumes.

### **Step-by-Step Process:**
1. **Deploy the Model in Asynchronous Mode:**
   ```python
   from sagemaker import Model
   model = Model(model_data='s3://models/model.tar.gz')
   async_predictor = model.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large',
       async_inference_config={
           "OutputPath": "s3://async-output/"
       }
   )
   ```

2. **Invoke Asynchronous Endpoint with Large Payload:**
   ```python
   response = async_predictor.predict(
       data=input_data,
       accept="application/json",
       content_type="application/json"
   )
   ```

3. **Retrieve Predictions from S3 Output Location:**
   ```python
   import boto3
   s3 = boto3.client('s3')
   response = s3.get_object(Bucket='async-output', Key='result.json')
   print(response['Body'].read().decode('utf-8'))
   ```

---

## 4. Running ML Inference with AWS Lambda & API Gateway

AWS Lambda enables serverless ML inference, reducing costs by running predictions only when needed.

### **Step-by-Step Process:**
1. **Convert ML Model for AWS Lambda Deployment:**
   - Optimize the model size and convert it into a lightweight format like ONNX or TensorFlow Lite.

2. **Deploy the Model in AWS Lambda:**
   ```python
   import json
   import boto3
   import numpy as np
   
   def lambda_handler(event, context):
       model = boto3.client('sagemaker-runtime')
       input_data = json.loads(event['body'])
       response = model.invoke_endpoint(
           EndpointName='ml-endpoint',
           ContentType='application/json',
           Body=json.dumps(input_data)
       )
       return {
           'statusCode': 200,
           'body': response['Body'].read().decode()
       }
   ```

3. **Expose Lambda Model via API Gateway:**
   - Set up an API Gateway to trigger the Lambda function.
   - Configure authentication using IAM roles or AWS Cognito.

---

## 5. Example: Deploying a Fraud Detection API Using SageMaker

### **Scenario:**
A bank wants to detect fraudulent transactions in real-time using a machine learning model.

### **Step-by-Step Implementation:**
1. **Train a Fraud Detection Model with SageMaker:**
   ```python
   from sagemaker.xgboost import XGBoost
   estimator = XGBoost(role=role, instance_count=1, instance_type='ml.m5.large')
   estimator.fit({'train': 's3://fraud-detection-data/train/'})
   ```

2. **Deploy the Model as a SageMaker Endpoint:**
   ```python
   predictor = estimator.deploy(instance_type='ml.m5.large', initial_instance_count=1)
   ```

3. **Set Up an API Gateway to Trigger Inference Requests:**
   - Create an API Gateway REST API.
   - Link it to a Lambda function that invokes the SageMaker Endpoint.

4. **Process Real-Time Fraud Detection Requests:**
   ```python
   import requests
   fraud_request = {'transaction_amount': 5000, 'location': 'NY', 'time': '12:00'}
   response = requests.post("https://api.example.com/fraud-detection", json=fraud_request)
   print(response.json())
   ```

5. **Monitor API Requests in CloudWatch:**
   - Track latency and errors using **CloudWatch Logs and Metrics**.

---

### **Conclusion**
AWS provides a flexible and scalable infrastructure for both **batch processing and real-time inference**. Services like **SageMaker Endpoints, Lambda, API Gateway, and Step Functions** help businesses deploy ML models for applications like fraud detection, customer segmentation, and recommendation engines.

