# Unsupervised Learning on AWS

## Table of Contents
1. [What is Unsupervised Learning?](#what-is-unsupervised-learning)
2. [Implementing Clustering Models (K-Means) on SageMaker](#implementing-clustering-models-k-means-on-sagemaker)
3. [Using Amazon Personalize for Recommender Systems](#using-amazon-personalize-for-recommender-systems)
4. [Anomaly Detection Using Amazon Lookout for Metrics](#anomaly-detection-using-amazon-lookout-for-metrics)
5. [Example: Customer Segmentation Using SageMaker K-Means](#example-customer-segmentation-using-sagemaker-k-means)

---

## What is Unsupervised Learning?

Unsupervised learning is a type of machine learning where an algorithm learns patterns from unlabeled data. Unlike supervised learning, where models are trained using labeled datasets, unsupervised learning identifies hidden structures and patterns without explicit labels.

### **Common Types of Unsupervised Learning:**
- **Clustering:** Grouping similar data points together (e.g., customer segmentation, image grouping).
- **Dimensionality Reduction:** Reducing the number of input features while preserving important information.
- **Anomaly Detection:** Identifying unusual patterns that do not conform to expected behavior.
- **Association Rule Learning:** Finding relationships between variables (e.g., market basket analysis).

AWS provides multiple services to implement unsupervised learning, such as **Amazon SageMaker, Amazon Personalize, and Amazon Lookout for Metrics**.

---

## Implementing Clustering Models (K-Means) on SageMaker

### **Theory**
Clustering is a technique used to group similar data points together based on their features. **K-Means Clustering** is a popular algorithm that assigns data points into K clusters by minimizing the distance between points within a cluster.

### **Step-by-Step Process:**
1. **Prepare the Dataset:**
   - Upload an unlabeled dataset to Amazon S3.
   - Ensure numerical features are standardized.
2. **Create a SageMaker Training Job:**
   - Use the built-in K-Means algorithm in SageMaker.
   ```python
   from sagemaker import KMeans
   kmeans = KMeans(role=role, instance_count=1, instance_type='ml.m5.large', k=5)
   kmeans.fit({'train': training_data_s3})
   ```
3. **Deploy the Model:**
   - Deploy the trained clustering model as a SageMaker Endpoint.
   ```python
   predictor = kmeans.deploy(instance_type='ml.m5.large', initial_instance_count=1)
   ```
4. **Make Predictions:**
   ```python
   cluster_assignments = predictor.predict(sample_data)
   print(cluster_assignments)
   ```

---

## Using Amazon Personalize for Recommender Systems

### **Theory**
Amazon Personalize is a managed service that builds recommendation systems using collaborative filtering and deep learning techniques. It helps in providing personalized product recommendations, content suggestions, and customer engagement.

### **Step-by-Step Process:**
1. **Prepare User Interaction Data:**
   - Upload historical user interaction data (e.g., clicks, purchases) to Amazon S3.
   - Ensure data includes user IDs, item IDs, and timestamps.
2. **Create a Dataset Group in Amazon Personalize:**
   ```python
   personalize = boto3.client('personalize')
   response = personalize.create_dataset_group(name='recommendation-group')
   ```
3. **Train a Recommendation Model:**
   - Choose a predefined algorithm such as **User-Personalization** or **Popularity-Based Ranking**.
   ```python
   response = personalize.create_solution(
       name='user-recommendation',
       datasetGroupArn='recommendation-group',
       recipeArn='arn:aws:personalize:::recipe/aws-user-personalization'
   )
   ```
4. **Deploy the Model and Generate Recommendations:**
   ```python
   response = personalize.get_recommendations(
       campaignArn='recommendation-campaign',
       userId='12345'
   )
   print(response)
   ```

---

## Anomaly Detection Using Amazon Lookout for Metrics

### **Theory**
Anomaly detection involves identifying deviations from normal behavior. **Amazon Lookout for Metrics** applies machine learning to detect anomalies in time-series data, such as sales trends, fraud detection, and system monitoring.

### **Step-by-Step Process:**
1. **Prepare and Upload Time-Series Data:**
   - Store structured time-series data (e.g., sales records, sensor data) in Amazon S3.
2. **Create an Anomaly Detection Detector:**
   ```python
   lookout = boto3.client('lookoutmetrics')
   response = lookout.create_anomaly_detector(
       AnomalyDetectorName='sales-anomalies',
       AnomalyDetectorDescription='Detect anomalies in sales trends',
       AnomalyDetectorConfig={'AnomalyDetectorFrequency': 'DAILY'}
   )
   ```
3. **Train the Anomaly Detection Model:**
   ```python
   response = lookout.create_alert(
       AlertName='SalesAlert',
       AnomalyDetectorArn='sales-anomalies',
       AlertSensitivityThreshold=75
   )
   ```
4. **Monitor and Analyze Anomalies:**
   - Configure Amazon CloudWatch for automated alerts on detected anomalies.

---

## Example: Customer Segmentation Using SageMaker K-Means

### **Theory**
Customer segmentation is the process of dividing a customer base into groups based on common characteristics, such as purchase behavior, demographics, or preferences. **K-Means clustering** is an effective method for segmenting customers into meaningful groups.

### **Step-by-Step Process:**
1. **Prepare Customer Data:**
   - Gather customer purchase history, browsing behavior, and demographics.
   - Store data in Amazon S3 in a CSV format.
2. **Preprocess Data:**
   - Use AWS Glue or Pandas to normalize numerical fields and encode categorical variables.
   ```python
   df = pd.get_dummies(df, columns=['location', 'purchase_category'])
   df.fillna(df.mean(), inplace=True)
   ```
3. **Train a K-Means Model:**
   ```python
   from sagemaker import KMeans
   kmeans = KMeans(role=role, instance_count=1, instance_type='ml.m5.large', k=4)
   kmeans.fit({'train': training_data_s3})
   ```
4. **Assign Customers to Clusters:**
   ```python
   predictor = kmeans.deploy(instance_type='ml.m5.large', initial_instance_count=1)
   customer_segments = predictor.predict(new_customer_data)
   print(customer_segments)
   ```
5. **Analyze Customer Segments:**
   - Use SageMaker Clarify for feature importance analysis.
   - Apply Amazon QuickSight for visualizing customer segmentation insights.

---

### **Conclusion**
Unsupervised learning enables automatic pattern discovery in data. AWS services such as SageMaker (for clustering), Amazon Personalize (for recommendations), and Lookout for Metrics (for anomaly detection) provide scalable solutions for implementing these models. Businesses can leverage these techniques for tasks like **customer segmentation, fraud detection, and personalized recommendations**.

