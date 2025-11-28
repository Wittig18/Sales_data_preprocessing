# Sales_data_preprocessing

Data Preprocessing for Machine Learning — Beginner Project

This project demonstrates a full end-to-end data preprocessing pipeline for preparing a dirty dataset for machine learning. It covers all essential steps beginners must understand before building ML models.

**Project Overview**
Real-world datasets are rarely clean. They contain:

Missing values

Wrong data types

Outliers

Inconsistent categories

Skewed distributions

Duplicate entries

This project walks through cleaning and transforming a messy dataset to make it suitable for machine learning.

**What This Project Covers**

1. Importing Dependencies

Using essential libraries like:

pandas

numpy

sklearn.preprocessing

sklearn.impute

sklearn.compose

sklearn.pipeline

2. Exploring the Raw Dataset

We check for:

Missing values

Data types

Unique values in categorical columns

Outliers

Skewed features

Summary statistics

3. Handling Missing Values

Numerical columns

Filled using median imputation

Categorical columns

Filled using the mode (most common category)

Dates

Converted to datetime

Missing dates replaced with median date

4. Fixing Wrong Data Types

5. Handling Outliers

Using percentile capping:

for col in ["Income", "PurchaseAmount"]:
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower, upper)

6. Creating New Features (Feature Engineering)

Converting dates to duration

Creating bin

7. Encoding Categorical Variables

Two methods were covered:

✔ OneHotEncoder (sklearn)

✔ Pandas get_dummies() for beginners

8. Scaling Numerical Features

Using StandardScaler

This allows automatic preprocessing when training any ML model.

**Why This Project Matters**
Proper data preprocessing:

Improves model accuracy

Prevents errors in training

Makes ML pipelines robust

Ensures reproducibility

Prepares beginners for real-world datasets

Machine learning is 80% data cleaning and 20% modeling — so mastering this is essential.

**Trained models** 
Linear Regression = r2_score: 0.46

**Hyperparametilizing**
RandomForestRegressor = r2 = 0.6 

