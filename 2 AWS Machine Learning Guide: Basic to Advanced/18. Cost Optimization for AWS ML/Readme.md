# Cost Optimization for AWS ML

## Table of Contents
1. [Understanding AWS Pricing for ML Workloads](#understanding-aws-pricing-for-ml-workloads)
2. [AWS Free Tier ML Resources](#aws-free-tier-ml-resources)
3. [Reducing ML Costs with Spot Instances](#reducing-ml-costs-with-spot-instances)
4. [Optimizing SageMaker Training with Managed Spot Training](#optimizing-sagemaker-training-with-managed-spot-training)
5. [Example: Optimizing ML Costs with SageMaker Savings Plans](#example-optimizing-ml-costs-with-sagemaker-savings-plans)

---

## 1. Understanding AWS Pricing for ML Workloads
AWS provides multiple pricing models to optimize ML workloads based on usage patterns and cost efficiency.

### **AWS ML Pricing Models:**
- **On-Demand Instances** – Pay for compute power as needed.
- **Spot Instances** – Get up to 90% cost savings for interruptible workloads.
- **Reserved Instances** – Discounted pricing for long-term commitments.
- **SageMaker Savings Plans** – Cost-efficient plans for SageMaker usage.
- **AWS Free Tier** – Free access to select ML resources for new users.

### **Key Cost Factors:**
- **Compute Costs:** EC2, SageMaker, GPU vs. CPU instances.
- **Storage Costs:** S3, EBS, and database storage for ML datasets.
- **Data Transfer Fees:** Moving data across AWS services may incur costs.
- **Inference Costs:** Costs associated with running ML models in production.

---

## 2. AWS Free Tier ML Resources
AWS provides free-tier resources to help users experiment with ML at no cost.

### **UI Steps:**
1. **Navigate to the AWS Free Tier Page:**
   - Go to [AWS Free Tier](https://aws.amazon.com/free/).
2. **Check Free ML Services:**
   - Look for **SageMaker**, **Lambda**, **Comprehend**, and **Rekognition** free-tier offers.
3. **Set Up Billing Alerts:**
   - Use AWS Cost Explorer to track free-tier usage and avoid overages.

### **CLI Steps:**
1. **Check AWS Free Tier Usage:**
   ```bash
   aws ce get-cost-and-usage --time-period Start=2025-01-01,End=2025-01-31 --granularity MONTHLY
   ```
2. **Monitor SageMaker Free Tier Usage:**
   ```bash
   aws sagemaker list-training-jobs
   ```

### **Common Free-Tier ML Resources:**
- **Amazon SageMaker:** 250 hours/month of `ml.t3.medium` notebook instances.
- **AWS Lambda:** 1 million free requests/month for ML inference.
- **Amazon Rekognition:** 5,000 free image analyses/month.
- **Amazon Comprehend:** 50,000 free text analyses/month.
- **Amazon Forecast:** 1,000 free inference hours/month.

---

## 3. Reducing ML Costs with Spot Instances
AWS Spot Instances allow users to run ML workloads at significantly lower costs than On-Demand instances.

### **UI Steps:**
1. **Go to AWS EC2 Console:**
   - Navigate to **EC2 Dashboard** → **Spot Requests**.
2. **Request a Spot Instance:**
   - Click **Request Spot Instances**.
   - Choose an instance type (e.g., `ml.p3.2xlarge`).
3. **Monitor Spot Usage in AWS Cost Explorer.**

### **CLI Steps:**
1. **Launch a Spot Instance for ML Training:**
   ```bash
   aws ec2 request-spot-instances --instance-type ml.p3.2xlarge --spot-price 0.20
   ```
2. **Monitor Spot Instance Availability:**
   ```bash
   aws ec2 describe-spot-instance-requests
   ```

---

## 4. Optimizing SageMaker Training with Managed Spot Training
Amazon SageMaker provides **Managed Spot Training**, allowing up to 90% cost savings compared to on-demand training jobs.

### **UI Steps:**
1. **Open SageMaker Studio:**
   - Navigate to the **SageMaker Console** → **Training Jobs**.
2. **Create a Training Job with Spot Instances:**
   - Choose **Use Managed Spot Training**.
   - Set a checkpoint location in S3 to save training progress.
3. **Monitor Job Progress in SageMaker Studio.**

### **CLI Steps:**
1. **Enable Managed Spot Training:**
   ```python
   from sagemaker import Estimator
   
   estimator = Estimator(
       image_uri="xgboost:latest",
       role=role,
       instance_count=1,
       instance_type="ml.p3.2xlarge",
       use_spot_instances=True,
       max_wait=3600,
       max_run=1800
   )
   ```
2. **Monitor Cost Savings in SageMaker Studio:**
   - Track cost savings in the **SageMaker Experiment Tracker**.

3. **Use Checkpointing for Long-Running Jobs:**
   ```python
   estimator.fit(inputs, checkpoint_s3_uri='s3://model-checkpoints/')
   ```

---

## 5. Example: Optimizing ML Costs with SageMaker Savings Plans
SageMaker Savings Plans offer **up to 64% savings** on SageMaker usage with committed spending plans.

### **UI Steps:**
1. **Go to AWS Cost Management Console:**
   - Navigate to **AWS Cost Explorer** → **SageMaker Savings Plans**.
2. **Select a Commitment Plan:**
   - Choose a **1-year or 3-year** commitment based on usage predictions.
3. **Monitor Cost Savings Over Time.**

### **CLI Steps:**
1. **Enable a SageMaker Savings Plan:**
   ```bash
   aws savingsplans create-savings-plan --savings-plan-type SageMaker --commitment 100 --duration 31536000
   ```
2. **Monitor Savings and Adjust as Needed:**
   ```bash
   aws ce get-savings-plans-utilization
   ```

---

### **Conclusion**
AWS provides multiple ways to **optimize ML costs**, from using **Spot Instances and Managed Spot Training** to leveraging **SageMaker Savings Plans** and **AWS Free Tier** resources. By implementing cost-efficient strategies, businesses can maximize their ML investments while minimizing expenses.

