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
