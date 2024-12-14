# README for Time Series Data Processing and Prediction Scripts

## Overview

This repository contains three Python scripts designed for processing time series data and predicting future values using regression techniques. The scripts read data from a CSV file, preprocess it, and apply machine learning models (specifically Random Forest regressor) to forecast values at different time steps: 1, 30, and 60 time steps ahead.

### Scripts:
- `data_process.py`: Processes the data and predicts the next value based on the most recent data.
- `data_process_30.py`: Processes the data to predict values 30 steps ahead.
- `data_process_60.py`: Processes the data to predict values 60 steps ahead.
- `data_process_90.py`: Processes the data to predict values 90 steps ahead.
- `data_process_h.py`: Predicts values 10 steps ahead using a Support Vector Regressor (SVR).
- `data_process_h_m.py`: Compares predictions from multiple regression models (RF, Lasso, KNN, SVR, AdaBoost, and Gradient Boosting) for 10 steps ahead.

## Requirements

To run these scripts, ensure you have Python installed along with the following packages:

- numpy
- matplotlib
- scikit-learn
- openpyxl

To install the required packages, you can use pip:

```bash
pip install numpy matplotlib scikit-learn openpyxl
```

## Running the Scripts

```bash
python data_process.py
python data_process_30.py
python data_process_60.py
python data_process_90.py
python data_process_h.py
python data_process_h_m.py
```