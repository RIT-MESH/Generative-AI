# Data Preparation and Management

## Table of Contents
1. [Importing and Storing Data in Amazon S3](#importing-and-storing-data-in-amazon-s3)
2. [Data Wrangling with AWS Glue & SageMaker Data Wrangler](#data-wrangling-with-aws-glue-sagemaker-data-wrangler)
3. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
4. [Feature Engineering with SageMaker Feature Store](#feature-engineering-with-sagemaker-feature-store)
5. [Handling Missing Data in AWS ML Pipelines](#handling-missing-data-in-aws-ml-pipelines)
6. [Creating Scalable Data Lakes with AWS Lake Formation](#creating-scalable-data-lakes-with-aws-lake-formation)

---

## Importing and Storing Data in Amazon S3
Amazon S3 is a scalable storage solution for storing datasets used in machine learning.

### **Step-by-Step Process:**
1. **Create an S3 Bucket:**
   - Navigate to **Amazon S3 Console**.
   - Click **Create bucket**.
   - Enter a unique bucket name (e.g., `ml-datasets-bucket`).
   - Select the appropriate AWS region.
2. **Set Permissions and Policies:**
   - Define access control settings and IAM permissions.
   - Use **Amazon S3 Object Lock** for data protection.
3. **Upload Data:**
   - Click **Upload**, then select files or folders.
   - Enable server-side encryption for data security.
4. **Organize Data:**
   - Use folders (prefixes) to manage structured data.
   - Enable **S3 Lifecycle Policies** to automate data archival.
5. **Access Data Programmatically:**
   - Use **Boto3 SDK**:
     ```python
     import boto3
     s3 = boto3.client('s3')
     s3.upload_file('localfile.csv', 'ml-datasets-bucket', 'datasets/localfile.csv')
     ```

---

## Data Wrangling with AWS Glue & SageMaker Data Wrangler
AWS Glue and SageMaker Data Wrangler enable automated data transformation and preparation.

### **Step-by-Step Process:**
1. **Use AWS Glue for ETL:**
   - Navigate to **AWS Glue Console**.
   - Create a **Glue Crawler** to catalog data.
   - Define an **ETL job** using PySpark or AWS Glue Studio.
   - Run transformations and load data into Amazon S3 or Redshift.
2. **Use SageMaker Data Wrangler for Advanced Data Processing:**
   - Open SageMaker Studio and select **Data Wrangler**.
   - Import data from S3, Redshift, or Snowflake.
   - Perform exploratory data analysis and feature engineering.
   - Export the processed dataset to Amazon S3 or SageMaker Feature Store.

---

## Data Cleaning and Preprocessing
Cleaning data ensures better model accuracy and removes inconsistencies.

### **Step-by-Step Process:**
1. **Identify Missing or Corrupt Data:**
   - Load the dataset using Pandas or AWS Data Wrangler.
     ```python
     import pandas as pd
     df = pd.read_csv('s3://ml-datasets-bucket/datasets/data.csv')
     df.info()
     ```
2. **Handle Missing Data:**
   - Fill missing values using mean imputation:
     ```python
     df.fillna(df.mean(), inplace=True)
     ```
   - Drop rows with missing values:
     ```python
     df.dropna(inplace=True)
     ```
3. **Normalize and Standardize Data:**
   - Scale numerical features using MinMaxScaler or StandardScaler.
   - Encode categorical variables for ML training.
4. **Store Cleaned Data in S3 or Redshift:**
   - Save the processed dataset:
     ```python
     df.to_csv('s3://ml-datasets-bucket/processed-data.csv', index=False)
     ```

---

## Feature Engineering with SageMaker Feature Store
Feature engineering helps in transforming raw data into meaningful features.

### **Step-by-Step Process:**
1. **Set Up SageMaker Feature Store:**
   - Navigate to **SageMaker Feature Store** in AWS Console.
   - Create a feature group to store transformed data.
2. **Ingest Data into Feature Store:**
   - Format data as a DataFrame and write to Feature Store:
     ```python
     import sagemaker.feature_store as fs
     feature_group = fs.FeatureGroup(name='customer-features', sagemaker_session=session)
     feature_group.ingest(data_frame=df, max_workers=3, wait=True)
     ```
3. **Retrieve Features for Model Training:**
   - Query features using Athena or SageMaker SDK:
     ```python
     query = feature_group.athena_query()
     query.run()
     ```

---

## Handling Missing Data in AWS ML Pipelines
Missing data can negatively impact model performance and must be handled systematically.

### **Step-by-Step Process:**
1. **Detect Missing Data:**
   - Check missing values in AWS Glue Data Catalog or SageMaker Data Wrangler.
2. **Apply Strategies to Handle Missing Data:**
   - Mean/Median imputation for numerical fields.
   - Mode imputation for categorical variables.
   - Use AWS Glue DynamicFrames for structured missing data processing.
3. **Integrate Data Cleaning in ML Pipelines:**
   - Automate preprocessing using **SageMaker Processing Jobs**.
   - Store cleaned datasets in Amazon S3 for training models.

---

## Creating Scalable Data Lakes with AWS Lake Formation
AWS Lake Formation enables creating a centralized and secure data lake for ML applications.

### **Step-by-Step Process:**
1. **Set Up AWS Lake Formation:**
   - Navigate to **AWS Lake Formation Console**.
   - Register an S3 bucket as a data lake.
2. **Define Security and Access Permissions:**
   - Use **IAM policies** and **Lake Formation permissions** to control access.
   - Enable **row-level security** to restrict access to specific datasets.
3. **Catalog and Structure Data:**
   - Use **Glue Crawlers** to index data from different sources.
   - Create tables in **AWS Glue Data Catalog**.
4. **Query Data Efficiently:**
   - Use Amazon Athena or Redshift Spectrum to run SQL queries on data stored in S3.
5. **Automate Data Ingestion and Transformation:**
   - Set up AWS Glue ETL jobs to load data into the lake.
   - Automate pipeline execution using AWS Step Functions.

---

### **Conclusion**
By implementing these AWS services, organizations can efficiently manage data preparation, cleaning, feature engineering, and storage. These steps ensure scalable, high-performance ML pipelines with optimized data processing workflows.

