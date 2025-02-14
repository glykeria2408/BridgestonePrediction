# BridgestonePrediction
# 📌 Project Overview

This project aims to analyze key operational and financial factors affecting **Bridgestone's monthly revenue** using **machine learning techniques**. By integrating data related to **production output, defective rates, supply chain delays, energy consumption, AI-driven optimizations, marketing spend, and employee count**, we seek to uncover insights that can optimize **revenue forecasting and operational efficiency**.

## 📂 Dataset Information
We utilize a comprehensive dataset containing **1,200 monthly observations** with the following key variables:

1. **Operational Data**
   - **Production Output**: Number of tires produced per month.
   - **Defective Rate**: Percentage of defective tires in production.
   - **Supply Chain Delays**: Number of days of delay in receiving raw materials.
   - **Raw Material Cost**: Monthly expenditure on raw materials.
   - **Energy Consumption**: Total electricity used in kWh.
   - **AI-Optimized Production**: Whether AI was used in production (0 = No, 1 = Yes).
   
2. **Financial & Workforce Data**
   - **Marketing Spend**: Monthly expenditure on marketing campaigns.
   - **Employee Count**: Number of employees working in production.
   - **Revenue**: Monthly revenue generated (target variable for prediction).

## 🚀 Workflow

### Step 1: Load the Datasets
We start by reading the dataset into a **Pandas dataframe**, inspecting it for **missing values** and **inconsistencies**.

### Step 2: Data Wrangling & Cleaning
- **Fixing inconsistencies** in categorical variables (e.g., correcting AI-Optimized Production values).
- **Handling missing values** using **mean imputation**.
- **Standardizing numerical features** to improve model performance.

### Step 3: Feature Engineering
- **Extracting time-based features** from the month column.
- **Transforming categorical data** for better predictive modeling.

### Step 4: Predicting Monthly Revenue
- Implementing a **Random Forest Regressor** to estimate revenue based on operational factors.
- **Splitting the data** into training and testing sets.
- **Evaluating model performance** using **R² score**.

## 📊Power BI Visualizations
1. **Revenue vs. Marketing Spend** - A scatter plot to analyze the correlation between marketing investment and revenue growth.
2. **Production Output vs. Revenue** - A line or bar chart comparing production levels to monthly revenue trends.
3. **AI-Driven Production Impact** - A stacked bar chart showing revenue performance for AI-optimized vs. non-AI production months.

This project serves as a **data-driven foundation for optimizing operational decisions** and improving **revenue forecasting** at Bridgestone. 🚀


