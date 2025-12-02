Netflix Movie Analysis using Time Series

A data analysis project that explores Netflix’s movie release trends, viewer patterns, and platform growth using time series techniques. This project uses Python, Pandas, Matplotlib/Seaborn, and statistical modeling to derive meaningful insights from Netflix’s movie dataset.
Project Overview

This project focuses on analyzing Netflix movies using time series analysis to understand:

Yearly and monthly trends in content release
Popular genres over time
Distribution of movie duration across years
Forecasting movie releases (using ARIMA/Prophet)
Seasonal patterns and long-term trends

The goal is to use time series visualization and forecasting models to reveal patterns in Netflix’s catalog evolution.

Dataset

Source: Netflix Movies & TV Shows dataset (e.g., Kaggle)

Key Columns Used:
title
type
release_year
date_added
duration
listed_in (genre)

Languages: Python

Libraries:

Pandas
NumPy
Matplotlib / Seaborn
Plotly (optional)
Statsmodels / Prophet (for forecasting)
Tools: Jupyter Notebook / Google Colab

Key Steps in the Project

✔️ 1. Data Cleaning & Preprocessing

Convert date fields to datetime format
Handle missing values
Extract year, month, and day
Filter for only movies

✔️ 2. Exploratory Data Analysis (EDA)

Trend of movies released by year
Monthly content addition trends
Duration distribution
Genre analysis

✔️ 3. Time Series Creation

Aggregate movies added per month
Create univariate time series
Check stationarity (ADF Test)
Decompose time series into:
Trend
Seasonality
Residual

✔️ 4. Time Series Forecasting

Build models like:
ARIMA/SARIMA
Facebook Prophet (optional)
Forecast movie additions for the next 12–24 months

✔️ 5. Visualization

Line plots for trends
Heatmaps for monthly patterns
Seasonal decomposition plots
Actual vs. Forecast graphs






Project Overview – Customer Churn Prediction (Deep Learning)

This project focuses on building a Deep Learning–based Customer Churn Prediction model to help businesses identify customers who are likely to leave the platform. Churn prediction is a critical task for subscription-based companies, telecom providers, banks, and online services because retaining customers is far more cost-effective than acquiring new ones.

Using a neural network built with TensorFlow/Keras, this project analyzes customer behavior, usage patterns, and demographic features to predict whether a customer will churn. The model aims to support data-driven customer retention strategies.


🎯 Objectives

Understand customer behavior and patterns leading to churn
Perform data preprocessing, EDA, and feature engineering
Build and train a Deep Learning classification model
Evaluate model performance using accuracy, precision, recall, F1-score, ROC curve, and confusion matrix
Interpret feature importance (via SHAP / permutation importance)
Provide actionable business insights to reduce churn

tech

Python

TensorFlow/Keras – Deep learning model
Pandas, NumPy – Data manipulation
Matplotlib, Seaborn – Visualizations
Scikit-learn – Preprocessing & metrics

Key Steps Included in the Project

✔️ 1. Data Preprocessing

Handling missing values
Encoding categorical variables
Feature scaling
Train-test split

✔️ 2. Exploratory Data Analysis (EDA)

Churn distribution analysis
Correlation heatmap
Customer demographics and behavior trends
Visualization of key factors influencing churn

✔️ 3. Deep Learning Model Development

Build a fully connected neural network
Implement dropout & batch normalization
Hyperparameter tuning (layers, neurons, activation functions)
Model training & validation

✔️ 4. Model Evaluation

Confusion matrix
Classification report
ROC-AUC Curve
Precision–Recall Curve
Comparison with baseline ML models (optional)

✔️ 5. Insights & Recommendations

Example insights to include:

High churn among customers with low monthly usage
Customers with lower tenure are more likely to churn
Contract type significantly affects retention
Model can help customer support teams target “at-risk” users
