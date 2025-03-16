# AWS ML Services Overview

## Table of Contents
1. [Setting Up AWS ML Environment](#setting-up-aws-ml-environment)
   - [Creating an AWS Account](#1-creating-an-aws-account)
   - [Setting Up IAM Roles for ML Workloads](#2-setting-up-iam-roles-for-ml-workloads)
   - [Configuring S3 Storage for Datasets](#3-configuring-s3-storage-for-datasets)
   - [Using AWS Cloud9 for ML Development](#4-using-aws-cloud9-for-ml-development)
   - [Setting Up SageMaker Notebook Instances](#5-setting-up-sagemaker-notebook-instances)
   - [Installing AWS SDK (Boto3) for Python](#6-installing-aws-sdk-boto3-for-python)

## Setting Up AWS ML Environment
Setting up the AWS ML environment involves creating an AWS account, configuring IAM roles, setting up storage, using development environments, and installing required SDKs.

### **Step-by-Step Process:**

### **1. Creating an AWS Account:**
1. **Go to AWS Sign-Up Page:**
   - Visit [AWS Website](https://aws.amazon.com/) and click on **Create an AWS Account**.
2. **Enter Account Details:**
   - Provide an email address, choose an AWS account name, and set up a password.
3. **Select Account Type:**
   - Choose between **Personal** or **Professional**.
4. **Enter Billing Information:**
   - Provide credit/debit card details for billing verification.
5. **Verify Identity:**
   - Enter a phone number for OTP verification.
6. **Choose Support Plan:**
   - Select from **Basic**, **Developer**, or **Enterprise** support plans.
7. **Sign in to AWS Console:**
   - Log in to the AWS Management Console using root credentials.

---

### **2. Setting Up IAM Roles for ML Workloads**
1. **Access IAM Console:**
   - Go to the AWS IAM service in the Management Console.
2. **Create a New IAM Role:**
   - Click **Roles → Create Role**.
3. **Attach Policies:**
   - Select **AmazonS3FullAccess**, **AmazonSageMakerFullAccess**, and **AWSLambdaBasicExecutionRole** for ML workloads.
4. **Assign Role to Services:**
   - Choose AWS services like **SageMaker, Lambda, EC2** that will use this role.
5. **Create Role:**
   - Provide a role name (e.g., `SageMakerExecutionRole`) and click **Create Role**.
6. **Attach IAM Role to SageMaker Notebook:**
   - Navigate to SageMaker → Notebook Instances → Select Notebook → Attach IAM Role.

---

### **3. Configuring S3 Storage for Datasets**
1. **Open Amazon S3 Console:**
   - Navigate to **Amazon S3** in the AWS Management Console.
2. **Create a New Bucket:**
   - Click **Create Bucket**.
   - Enter a unique bucket name (e.g., `ml-dataset-storage`).
   - Choose the AWS Region where your ML workloads will run.
3. **Set Permissions:**
   - Enable **Bucket Versioning** for data recovery.
   - Apply IAM roles to restrict or allow access to specific users.
4. **Upload Datasets:**
   - Click **Upload** and add ML datasets from your local machine.
5. **Enable Logging and Monitoring:**
   - Use **S3 Access Logs** and **CloudTrail** for tracking data access.

---

### **4. Using AWS Cloud9 for ML Development**
1. **Open Cloud9 Console:**
   - Navigate to **AWS Cloud9** service.
2. **Create an Environment:**
   - Click **Create environment** and name it (`ml-dev-environment`).
   - Choose **Amazon EC2** as the host.
3. **Configure Instance:**
   - Select **Instance Type** (e.g., `t3.medium` for ML development).
   - Set auto-hibernate timeout.
4. **Launch Cloud9 IDE:**
   - Once the environment is created, open the Cloud9 IDE.
5. **Install Dependencies:**
   - Run:
     ```bash
     pip install boto3 pandas numpy scikit-learn
     ```
   - Install TensorFlow or PyTorch based on ML requirements.

---

### **5. Setting Up SageMaker Notebook Instances**
1. **Go to Amazon SageMaker Console:**
   - Navigate to **Amazon SageMaker → Notebook Instances**.
2. **Create a New Notebook:**
   - Click **Create notebook instance**.
   - Enter a name (`ml-notebook-instance`).
3. **Configure Instance Type:**
   - Select an instance type (`ml.t3.medium` for general ML or `ml.p3.2xlarge` for GPU workloads).
4. **Assign IAM Role:**
   - Attach the previously created IAM role (`SageMakerExecutionRole`).
5. **Enable Git Integration:**
   - Configure the notebook to sync with **GitHub or CodeCommit**.
6. **Launch Jupyter Notebook:**
   - Click **Open Jupyter** to start ML development.
7. **Install Required Libraries:**
   - Run:
     ```bash
     !pip install boto3 pandas numpy matplotlib
     ```

---

### **6. Installing AWS SDK (Boto3) for Python**
1. **Open Terminal (Cloud9, Local Machine, or SageMaker Notebook):**
   - If using local development, install Python 3.x.
2. **Install Boto3:**
   - Run the following command:
     ```bash
     pip install boto3
     ```
3. **Configure AWS CLI (Optional but Recommended):**
   - Run:
     ```bash
     aws configure
     ```
4. **Test Boto3 Installation:**
   - Open Python:
     ```python
     import boto3
     s3 = boto3.client('s3')
     print(s3.list_buckets())
     ```

---

### **Conclusion**
By following these steps, an AWS ML environment can be fully set up, allowing seamless ML development, storage, and deployment using AWS services. This setup ensures security, scalability, and ease of use for ML workloads.

