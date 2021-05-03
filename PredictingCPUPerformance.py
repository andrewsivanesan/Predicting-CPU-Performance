"""
Author - Andrew Sivanesan

Machine learning models for predicting CPU performance.

Uses the following custom module: CustomFuncs_CPU.py

"""

import os
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import make_pipeline, Pipeline
# from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU import train_single_model, train_multiple_models, test_model

##############################################
### Data preparation

wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
filename = "MachineData.txt"
cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
# read in data
df = pd.read_csv(wd + "\\" + filename, sep=",", names=cols)

# PRP has a strong positive and linear relationship with ERP
# other explanatory variables are strongly correlated with each other
# so we will use PRP as the sole explanatory variable
X = df["PRP"]
y = df["ERP"]

# 80% training set, 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# identify continous variables
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

# define transformer
# FixMe(20210503) - include outlier removal step in pipeline
X_transformer = ColumnTransformer(transformers=[("BoxCox", PowerTransformer(method="boxcox"), numeric_cols)])

##################################################
### Algorithm spot-check

train_multiple_models(X_train, y_train)