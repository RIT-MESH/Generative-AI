# Computer Vision on AWS

## Table of Contents
1. [Introduction to Amazon Rekognition](#introduction-to-amazon-rekognition)
2. [Face Detection and Object Recognition](#face-detection-and-object-recognition)
3. [Text Extraction from Images Using Amazon Textract](#text-extraction-from-images-using-amazon-textract)
4. [Real-World Case Study: License Plate Recognition System](#real-world-case-study-license-plate-recognition-system)
5. [Deploying Real-Time Object Detection Models with SageMaker](#deploying-real-time-object-detection-models-with-sagemaker)

---

## 1. Introduction to Amazon Rekognition
Amazon Rekognition is a fully managed computer vision service that enables image and video analysis through deep learning. It can perform face detection, object recognition, text extraction, and even content moderation.

### **Key Features of Amazon Rekognition:**
- **Face Detection & Analysis** – Identify faces, emotions, and demographic attributes.
- **Object & Scene Recognition** – Detect objects, scenes, and activities in images/videos.
- **Text Detection** – Extract text from images (OCR capability).
- **Content Moderation** – Identify inappropriate or restricted content.
- **Facial Recognition** – Compare faces across images for identity verification.
- **Custom Labels** – Train custom models for domain-specific object detection.

---

## 2. Face Detection and Object Recognition
Face detection and object recognition are core functionalities of Amazon Rekognition that enable AI-powered image analysis.

### **Step-by-Step Process for Face Detection:**
1. **Upload an Image to Amazon S3:**
   ```bash
   aws s3 cp face_image.jpg s3://rekognition-input-bucket/
   ```

2. **Use Amazon Rekognition for Face Detection:**
   ```python
   import boto3
   rekognition = boto3.client('rekognition')
   response = rekognition.detect_faces(
       Image={'S3Object': {'Bucket': 'rekognition-input-bucket', 'Name': 'face_image.jpg'}},
       Attributes=['ALL']
   )
   print(response)
   ```

3. **Analyze the Response:**
   - Extract detected face attributes such as age range, emotions, and facial landmarks.

### **Step-by-Step Process for Object Recognition:**
1. **Use Rekognition API to Detect Objects in an Image:**
   ```python
   response = rekognition.detect_labels(
       Image={'S3Object': {'Bucket': 'rekognition-input-bucket', 'Name': 'object_image.jpg'}},
       MaxLabels=10,
       MinConfidence=80
   )
   print(response)
   ```

2. **Analyze the Labels:**
   - Extract detected objects, their confidence scores, and scene descriptions.

---

## 3. Text Extraction from Images Using Amazon Textract
Amazon Textract is an AI-powered OCR service that extracts text from scanned documents, images, and PDFs.

### **Step-by-Step Process:**
1. **Upload a Document Image to Amazon S3:**
   ```bash
   aws s3 cp document.jpg s3://textract-input-bucket/
   ```

2. **Use Textract to Extract Text:**
   ```python
   textract = boto3.client('textract')
   response = textract.detect_document_text(
       Document={'S3Object': {'Bucket': 'textract-input-bucket', 'Name': 'document.jpg'}}
   )
   print(response)
   ```

3. **Extract and Structure the Data:**
   - Convert extracted text into structured data for further analysis.

---

## 4. Real-World Case Study: License Plate Recognition System

### **Workflow:**
1. **Capture License Plate Images Using a Camera System**
2. **Upload Images to Amazon S3 Automatically**
3. **Use Amazon Rekognition for Vehicle Detection:**
   ```python
   response = rekognition.detect_labels(
       Image={'S3Object': {'Bucket': 'license-plate-images', 'Name': 'car.jpg'}},
       MaxLabels=5
   )
   ```
4. **Use Amazon Textract for License Plate OCR:**
   ```python
   response = textract.detect_document_text(
       Document={'S3Object': {'Bucket': 'license-plate-images', 'Name': 'plate.jpg'}}
   )
   ```
5. **Store Extracted License Plate Data in a DynamoDB Table**

### **Storing Extracted License Plate Data in a DynamoDB Table:**
Amazon DynamoDB is a fully managed NoSQL database that provides fast and scalable storage for structured data. Storing extracted license plate data in DynamoDB allows for real-time lookups and integration with automated systems.

**Step-by-Step Process:**
1. **Create a DynamoDB Table for Storing License Plates:**
   ```python
   import boto3
   dynamodb = boto3.client('dynamodb')
   response = dynamodb.create_table(
       TableName='LicensePlates',
       KeySchema=[
           {'AttributeName': 'PlateNumber', 'KeyType': 'HASH'}
       ],
       AttributeDefinitions=[
           {'AttributeName': 'PlateNumber', 'AttributeType': 'S'}
       ],
       BillingMode='PAY_PER_REQUEST'
   )
   print("DynamoDB Table Created")
   ```

2. **Insert Extracted License Plate Data into DynamoDB:**
   ```python
   table = boto3.resource('dynamodb').Table('LicensePlates')
   plate_number = "XYZ1234"
   vehicle_details = {"Owner": "John Doe", "RegistrationState": "CA"}
   
   table.put_item(
       Item={
           'PlateNumber': plate_number,
           'VehicleDetails': vehicle_details
       }
   )
   print("License Plate Data Stored in DynamoDB")
   ```

6. **Trigger AWS Lambda for Automated Actions (e.g., Access Control, Toll Collection)**

### **Automating Actions Using AWS Lambda:**
AWS Lambda allows serverless execution of backend logic when a new entry is added to DynamoDB. This can be used for actions like automatic gate access control or toll collection.

**Step-by-Step Process:**
1. **Create a Lambda Function:**
   ```python
   import json
   import boto3
   
   def lambda_handler(event, context):
       dynamodb = boto3.resource('dynamodb')
       table = dynamodb.Table('LicensePlates')
       
       plate_number = event['PlateNumber']
       response = table.get_item(Key={'PlateNumber': plate_number})
       
       if 'Item' in response:
           action = "Access Granted" if response['Item']['RegistrationState'] == "CA" else "Access Denied"
           return {"statusCode": 200, "body": json.dumps({"PlateNumber": plate_number, "Action": action})}
       else:
           return {"statusCode": 404, "body": json.dumps({"Message": "Plate Not Found"})}
   ```

2. **Configure a DynamoDB Stream to Trigger the Lambda Function:**
   - Enable **DynamoDB Streams** in the **LicensePlates** table.
   - Configure an **Event Source Mapping** in AWS Lambda to trigger execution whenever a new record is added to DynamoDB.

3. **Automate Actions Based on Recognized License Plates:**
   - If the vehicle is registered for toll collection, automatically deduct charges from the linked account.
   - If the plate is authorized for a restricted area, send an API call to open a gate or send a notification.

---

By integrating **Amazon Rekognition, Textract, DynamoDB, and AWS Lambda**, this **License Plate Recognition System** enables real-time vehicle identification, secure access control, and automated toll collection.
License plate recognition (LPR) is a real-world application of AWS computer vision services that combines Amazon Rekognition and Amazon Textract.

### **Workflow:**
1. **Capture License Plate Images Using a Camera System**
2. **Upload Images to Amazon S3 Automatically**
3. **Use Amazon Rekognition for Vehicle Detection:**
   ```python
   response = rekognition.detect_labels(
       Image={'S3Object': {'Bucket': 'license-plate-images', 'Name': 'car.jpg'}},
       MaxLabels=5
   )
   ```
4. **Use Amazon Textract for License Plate OCR:**
   ```python
   response = textract.detect_document_text(
       Document={'S3Object': {'Bucket': 'license-plate-images', 'Name': 'plate.jpg'}}
   )
   ```
5. **Store Extracted License Plate Data in a DynamoDB Table**
6. **Trigger AWS Lambda for Automated Actions (e.g., Access Control, Toll Collection)**

---

## 5. Deploying Real-Time Object Detection Models with SageMaker
Amazon SageMaker enables real-time object detection with pre-trained deep learning models or custom-trained models.

### **Step-by-Step Process:**
1. **Prepare the Dataset:**
   - Store labeled image datasets in Amazon S3.

2. **Use SageMaker to Train an Object Detection Model:**
   ```python
   from sagemaker.tensorflow import TensorFlow
   estimator = TensorFlow(entry_point='train.py',
                          role=role,
                          instance_count=1,
                          instance_type='ml.p3.2xlarge',
                          framework_version='2.6')
   estimator.fit({'train': 's3://object-detection-dataset/train/'})
   ```

3. **Deploy the Model as a SageMaker Endpoint:**
   ```python
   predictor = estimator.deploy(instance_type='ml.m5.large', initial_instance_count=1)
   ```

4. **Use the Deployed Model for Real-Time Object Detection:**
   ```python
   response = predictor.predict(image_data)
   print(response)
   ```

---

### **Conclusion**
AWS provides powerful computer vision capabilities through Amazon Rekognition, Textract, and SageMaker. These services enable applications such as **face detection, text extraction, real-time object detection, and license plate recognition**. By leveraging these tools, businesses can build scalable AI-powered vision applications with minimal infrastructure overhead.

