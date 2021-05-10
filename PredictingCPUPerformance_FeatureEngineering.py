"""
Author - Andrew Sivanesan

Machine learning models for predicting CPU performance (multivariate model).

Uses the following custom module: CustomFuncs_CPU.py

"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,  KFold, cross_val_score
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import KBinsDiscretizer, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU import train_multiple_models
from CustomFuncs_CPU_KBins import test_model_KBins
from CustomFuncs_CPU_FeatureEng import train_multiple_models_KBins2, remove_outliers

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
numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
RemoveOutliers = FunctionTransformer(func=remove_outliers)

ct = ColumnTransformer(transformers=[("BoxCox", PowerTransformer(method="box-cox"), numeric_cols),
                                     ("RemoveOutliers", RemoveOutliers, numeric_cols)])
#FixMe(20210510) - apply RemoveOutliers to y_train as well as X_train
test_model_KBins(X_train, y_train, X_test, y_test, ABR_4, ct)



# train_multiple_models_KBins2(X_train, y_train, ABR_4, "quantile")
# train_multiple_models_KBins2(X_train, y_train, ABR_4, "kmeans")
# # best performance versus other strategies
# # bins = 6 
# train_multiple_models_KBins2(X_train, y_train, ABR_4, "uniform")

# ct = ColumnTransformer(transformers=[("Bins", 
#                                       KBinsDiscretizer(n_bins=9, encode='ordinal', strategy="uniform"), 
#                                       numeric_cols)])

