# Security & Compliance in AWS ML

## Table of Contents
1. [Setting Up IAM Roles and Permissions for ML](#setting-up-iam-roles-and-permissions-for-ml)
2. [Data Encryption & Compliance Best Practices](#data-encryption--compliance-best-practices)
3. [Securing ML Models with AWS PrivateLink](#securing-ml-models-with-aws-privatelink)
4. [Managing GDPR & HIPAA Compliance for AI Models](#managing-gdpr--hipaa-compliance-for-ai-models)
5. [Real-World Case: Secure ML Deployment in AWS](#real-world-case-secure-ml-deployment-in-aws)

---

## 1. Setting Up IAM Roles and Permissions for ML
AWS Identity and Access Management (IAM) allows you to manage permissions and access controls for ML models and services, ensuring **least privilege access** and compliance with security policies.

### **UI Steps:**
1. **Go to AWS IAM Console:**
   - Navigate to [IAM Console](https://console.aws.amazon.com/iam/).
   - Click **Roles** → **Create Role**.
2. **Select Trusted Entity:**
   - Choose **AWS Service** → **SageMaker**.
3. **Attach Required Policies:**
   - Attach `AmazonSageMakerFullAccess` and limit access to specific resources if needed.
4. **Review and Create Role:**
   - Name the role (e.g., `SageMakerExecutionRole`) and click **Create Role**.

### **CLI Steps:**
1. **Create an IAM Role for SageMaker Execution:**
   ```bash
   aws iam create-role --role-name SageMakerRole --assume-role-policy-document file://trust-policy.json
   ```
2. **Attach Required Permissions for SageMaker:**
   ```bash
   aws iam attach-role-policy --role-name SageMakerRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
   ```
3. **Grant Least Privilege Access for ML Pipelines:**
   - Restrict access to only necessary S3 buckets for training data.
   - Limit SageMaker model deployment to authorized users.

---

## 2. Data Encryption & Compliance Best Practices
AWS provides **encryption at rest and in transit** to secure ML models, training data, and inference outputs.

### **UI Steps:**
1. **Enable S3 Bucket Encryption:**
   - Open **Amazon S3 Console** → Select the bucket.
   - Click **Properties** → **Default Encryption**.
   - Choose **AWS Key Management Service (AWS KMS)** for encryption.
2. **Enable SageMaker Model Encryption:**
   - Navigate to **SageMaker Console** → **Training Jobs**.
   - Enable encryption by selecting a KMS key under **Output Data Configuration**.

### **CLI Steps:**
1. **Enable S3 Bucket Encryption for Training Data:**
   ```bash
   aws s3api put-bucket-encryption --bucket my-ml-bucket --server-side-encryption-configuration file://encryption-config.json
   ```
2. **Encrypt ML Models in SageMaker:**
   ```python
   from sagemaker import Model
   model = Model(
       model_data='s3://my-encrypted-models/model.tar.gz',
       role=role,
       enable_network_isolation=True
   )
   ```

---

## 3. Securing ML Models with AWS PrivateLink
AWS PrivateLink allows private connectivity between ML applications and AWS services without exposing traffic to the public internet.

### **UI Steps:**
1. **Create a VPC Endpoint for SageMaker:**
   - Go to **AWS VPC Console** → **Endpoints**.
   - Click **Create Endpoint** → Select **SageMaker API**.
   - Associate the endpoint with a private subnet.
2. **Modify Security Groups:**
   - Update **Security Groups** to allow traffic only from authorized services.

### **CLI Steps:**
1. **Create a VPC Endpoint for SageMaker:**
   ```bash
   aws ec2 create-vpc-endpoint --vpc-id vpc-abc123 --service-name com.amazonaws.us-east-1.sagemaker
   ```
2. **Restrict Access to ML Endpoints via PrivateLink:**
   ```json
   {
       "Effect": "Deny",
       "Action": "sagemaker:InvokeEndpoint",
       "Resource": "*",
       "Condition": {"StringNotEquals": {"aws:sourceVpce": "vpce-xyz456"}}
   }
   ```
3. **Monitor and Log ML Endpoint Activity Using AWS CloudTrail.**

---

## 4. Managing GDPR & HIPAA Compliance for AI Models
AWS provides built-in tools to help manage regulatory compliance requirements for AI applications.

### **UI Steps:**
1. **Enable AWS Audit Manager for Compliance Tracking:**
   - Navigate to **AWS Audit Manager** → **Create Assessment**.
   - Select **GDPR** or **HIPAA compliance framework**.
2. **Enable Data Protection for AI Models:**
   - Use **Amazon Macie** to scan for sensitive data in ML datasets.
   - Enable **S3 Object Lock** for regulatory data retention.

### **CLI Steps:**
1. **Enable GDPR & HIPAA Compliance Tracking:**
   ```bash
   aws auditmanager create-assessment --name GDPR_Audit --framework-id gdpr-framework-id
   ```
2. **Detect PII in ML Data Using Amazon Macie:**
   ```python
   import boto3
   macie = boto3.client('macie2')
   response = macie.create_classification_job(
       jobName='PII-Scan',
       s3JobDefinition={"BucketDefinitions": [{"BucketName": "ml-data-bucket"}]}
   )
   ```

---

## 5. Real-World Case: Secure ML Deployment in AWS

### **Scenario:**
A healthcare provider wants to deploy an AI-powered **medical diagnosis model** while ensuring **HIPAA compliance** and **data security**.

### **Implementation Steps:**
1. **Secure ML Training Data:**
   - Encrypt patient data stored in **Amazon S3**.
   - Use **AWS Macie** to detect PII.
2. **Restrict Model Access Using IAM & PrivateLink:**
   - Use **IAM roles** to control access to ML endpoints.
   - Deploy models within a **private VPC** using AWS PrivateLink.
3. **Enable Real-Time Compliance Monitoring:**
   - Set up **AWS CloudTrail** logs for all model inference requests.
   - Use **Amazon GuardDuty** to detect security threats.
4. **Automate Compliance Audits:**
   - Schedule audits using **AWS Audit Manager**.
   - Generate compliance reports for regulators.

---

### **Conclusion**
AWS provides powerful tools for securing ML models, protecting sensitive data, and ensuring compliance with GDPR, HIPAA, and enterprise security standards. By implementing IAM roles, encryption, AWS PrivateLink, and compliance best practices, businesses can **deploy ML securely at scale** while maintaining regulatory adherence.

