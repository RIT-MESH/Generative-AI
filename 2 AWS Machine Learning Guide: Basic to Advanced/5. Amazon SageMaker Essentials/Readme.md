# Amazon SageMaker Essentials

## Table of Contents
1. [Overview of SageMaker Architecture](#overview-of-sagemaker-architecture)
2. [Launching SageMaker Jupyter Notebooks](#launching-sagemaker-jupyter-notebooks)
3. [Built-in ML Algorithms in SageMaker](#built-in-ml-algorithms-in-sagemaker)
4. [Training ML Models with SageMaker Training Jobs](#training-ml-models-with-sagemaker-training-jobs)
5. [Hyperparameter Tuning with SageMaker Automatic Model Tuning](#hyperparameter-tuning-with-sagemaker-automatic-model-tuning)
6. [Deploying Models Using SageMaker Endpoints](#deploying-models-using-sagemaker-endpoints)
7. [Monitoring ML Models with SageMaker Model Monitor](#monitoring-ml-models-with-sagemaker-model-monitor)
8. [Debugging with SageMaker Debugger](#debugging-with-sagemaker-debugger)

---

## Overview of SageMaker Architecture
Amazon SageMaker provides a fully managed ML environment that integrates data preprocessing, model training, tuning, and deployment.

### **Key Components:**
- **SageMaker Studio**: An integrated development environment (IDE) for ML development.
- **Notebook Instances**: Managed Jupyter notebooks for code execution.
- **SageMaker Training Jobs**: Fully managed infrastructure for training ML models.
- **SageMaker Model Registry**: Centralized repository for trained models.
- **SageMaker Pipelines**: Automated ML workflows for CI/CD.
- **SageMaker Inference Endpoints**: Real-time model hosting and inference.

### **Workflow:**
1. **Data Preparation** → Store in S3
2. **Model Training** → Using SageMaker Training Jobs
3. **Model Evaluation & Tuning** → Hyperparameter tuning
4. **Model Deployment** → Deploy to SageMaker Endpoints
5. **Monitoring & Debugging** → Using Model Monitor & Debugger

---

## Launching SageMaker Jupyter Notebooks
SageMaker provides managed Jupyter notebooks for developing ML models.

### **Step-by-Step Process:**
1. **Navigate to SageMaker Console:**
   - Open AWS **SageMaker Console** → Click **Notebook Instances**.
2. **Create a Notebook Instance:**
   - Click **Create notebook instance**.
   - Choose an **Instance Type** (`ml.t3.medium` for general use, `ml.p3.2xlarge` for deep learning).
   - Assign an **IAM role** with `AmazonS3FullAccess` and `AmazonSageMakerFullAccess`.
3. **Launch Jupyter Notebook:**
   - Once created, click **Open Jupyter**.
   - Use built-in notebooks or create new ones.
4. **Install Required Libraries:**
   ```python
   !pip install pandas numpy scikit-learn boto3
   ```

---

## Built-in ML Algorithms in SageMaker
Amazon SageMaker provides pre-built ML algorithms optimized for distributed training.

### **Categories:**
1. **Supervised Learning:**
   - Linear Learner (Regression/Classification)
   - XGBoost (Gradient Boosting Trees)
   - Image Classification (ResNet, VGG)
2. **Unsupervised Learning:**
   - K-Means (Clustering)
   - Principal Component Analysis (Dimensionality Reduction)
3. **Anomaly Detection:**
   - Random Cut Forest (RCF)
4. **Time-Series Forecasting:**
   - DeepAR+

### **Example:** Train an XGBoost Model:
```python
import sagemaker
from sagemaker import get_execution_role

role = get_execution_role()
container = sagemaker.image_uris.retrieve('xgboost', region, 'latest')

estimator = sagemaker.estimator.Estimator(
    container,
    role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://ml-models/output')

estimator.fit({'train': 's3://ml-datasets/train.csv'})
```

---

## Training ML Models with SageMaker Training Jobs
SageMaker Training Jobs provide distributed infrastructure for scalable training.

### **Step-by-Step Process:**
1. **Prepare Data:**
   - Upload training data to Amazon S3.
2. **Define Training Parameters:**
   - Choose an ML algorithm.
   - Set hyperparameters.
3. **Launch Training Job:**
   ```python
   estimator.fit({'train': training_data_s3})
   ```
4. **Retrieve Model Artifacts:**
   - Stored in `s3://ml-models/output/`.

---

## Hyperparameter Tuning with SageMaker Automatic Model Tuning
Hyperparameter optimization automates finding the best parameters for a model.

### **Step-by-Step Process:**
1. **Define Hyperparameters to Tune:**
   ```python
   from sagemaker.tuner import HyperparameterTuner, IntegerParameter
   tuner = HyperparameterTuner(
       estimator,
       objective_metric_name='validation:rmse',
       hyperparameter_ranges={'num_round': IntegerParameter(10, 100)}
   )
   ```
2. **Launch Tuning Job:**
   ```python
   tuner.fit({'train': training_data_s3})
   ```
3. **Retrieve Best Model Parameters:**
   ```python
   best_job = tuner.best_training_job()
   ```

---

## Deploying Models Using SageMaker Endpoints
SageMaker Endpoints host models for real-time inference.

### **Step-by-Step Process:**
1. **Create Model Endpoint:**
   ```python
   predictor = estimator.deploy(
       initial_instance_count=1,
       instance_type='ml.m5.large'
   )
   ```
2. **Invoke Endpoint for Inference:**
   ```python
   response = predictor.predict(test_data)
   ```
3. **Monitor Inference Performance:**
   - Use CloudWatch logs for monitoring latency and errors.

---

## Monitoring ML Models with SageMaker Model Monitor
Model Monitor detects data drift and performance degradation.

### **Step-by-Step Process:**
1. **Enable Data Capture:**
   ```python
   from sagemaker.model_monitor import DataCaptureConfig
   data_capture = DataCaptureConfig(enable_capture=True, destination_s3_uri='s3://ml-monitoring/')
   ```
2. **Analyze Data Drift:**
   - Run automated monitoring jobs to compare live data with training data.

---

## Debugging with SageMaker Debugger
SageMaker Debugger helps identify training anomalies like vanishing gradients.

### **Step-by-Step Process:**
1. **Enable Debugger Hook:**
   ```python
   from sagemaker.debugger import DebuggerHookConfig
   debugger_config = DebuggerHookConfig(output_s3_uri='s3://debugger-logs/')
   ```
2. **Analyze Training Metrics:**
   - Use built-in rules to detect training failures.

---

### **Conclusion**
Amazon SageMaker provides a complete ML lifecycle, including development, training, tuning, deployment, and monitoring. Using SageMaker, ML models can be built efficiently with minimal infrastructure overhead, enabling scalable and automated ML solutions.

