"""
Author - Andrew Sivanesan

Machine learning models for predicting CPU performance (feature engineering).

Incorporate min, median and max values of PRP by vendor
Use Box-Cox transformation to fix skewness in predictors
Use robust scaling to address magnitude and range differences across predictors
K-bins discretisation not used (name of test_model_KBins function is relic of PredictingCPUPerformance_KBinsPRP.py)

No improvement on model 2 from PredictingCPUPerformance.py

Custom modules used: CustomFuncs_CPU_KBins.py

"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.compose import ColumnTransformer

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU_KBins import test_model_KBins

##############################################
### Data preparation

filename = "MachineData.txt"
cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
# read in data
df = pd.read_csv(wd + "\\" + filename, sep=",", names=cols)

pivot = (pd.pivot_table(df, index="vendor", values="PRP", aggfunc={"PRP": [np.min, np.median, np.max]})
            .rename(columns={"amin": "PRP_vendor_min", "amax": "PRP_vendor_max", "median": "PRP_vendor_med"})
            [["PRP_vendor_min", "PRP_vendor_med", "PRP_vendor_max"]])

df_join = df.merge(pivot, how="inner", on="vendor")

X = df_join[["PRP", "PRP_vendor_min", "PRP_vendor_med", "PRP_vendor_max"]]
y = df_join["ERP"]

# 80% training set, 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###############################################
### Evaluate model
ABR_4 = AdaBoostRegressor(random_state=42)
numeric_cols = list(X_train.select_dtypes(include=["int64", "float64"]).columns)

ct = ColumnTransformer(transformers=[("BoxCox", PowerTransformer(method="box-cox"), numeric_cols),
                                     ("Scaler", RobustScaler(), numeric_cols)])
test_model_KBins(X_train, y_train, X_test, y_test, ABR_4, ct)
# MSE of 6,907 (worse than model 2)