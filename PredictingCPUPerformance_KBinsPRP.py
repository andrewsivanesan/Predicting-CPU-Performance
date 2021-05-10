"""
Author - Andrew Sivanesan

Machine learning models for predicting CPU performance.

Uses the following custom module: CustomFuncs_CPU_KBins.py

"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostRegressor

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU_KBins import train_multiple_models_KBins, test_model_KBins, train_multiple_models_2

##############################################
### Data preparation

filename = "MachineData.txt"
cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
# read in data
df = pd.read_csv(wd + "\\" + filename, sep=",", names=cols)

X = df["PRP"]
y = df["ERP"]

# 80% training set, 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# reshape X_train and X_test for use in machine learning models
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

#############################################
### Feature engineering - discretising PRP variable
ABR_3 = AdaBoostRegressor(random_state=42)

train_multiple_models_KBins(X_train, y_train, ABR_3, "quantile")
train_multiple_models_KBins(X_train, y_train, ABR_3, "kmeans")
# best performance versus other strategies
# bins = 6 
train_multiple_models_KBins(X_train, y_train, ABR_3, "uniform")

# check whether AdaBoost regressor is still most performant model
ct = ColumnTransformer(transformers=[("Bins", 
                                      KBinsDiscretizer(n_bins=6, encode='ordinal', strategy="kmeans"), 
                                      [0])])
train_multiple_models_2(X_train, y_train, ct)
# AdaBoost regressor still most performant model (6 k-means bins on PRP)
test_model_KBins(X_train, y_train, X_test, y_test, ABR_3, ct)
# MSE of 3,078 (worse than model 2 in PredictingCPUPerformance.py)

