"""
Author - Andrew Sivanesan

Machine learning models for predicting CPU performance.

Uses the following custom module: CustomFuncs_CPU.py

"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU import train_multiple_models, test_model, grid_search

##############################################
### Data preparation

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
# reshape X_train and X_test for use in machine learning models
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
# define transformer
#ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
##################################################
### Algorithm spot-check

### Model 1 - benchmark model (naive mean prediction)
naive_mean_predictions = np.repeat(np.mean(X_train), len(X_test))
# calculate mean squared error
MSE = np.mean((naive_mean_predictions - X_test) ** 2)
print(MSE)
# MSE of 51,850 on the test set 
# Candidate models must have a lower MSE than this to be considered to have skill.

train_multiple_models(X_train, y_train)
# AdaBoost regressor has the best performance
# MSE of 3,865 
# All models outperform the benchmark model by a large margin

# make predictions on test set
ABR_1 = AdaBoostRegressor(random_state=42)
test_model(X_train, y_train, X_test, y_test, ABR_1)
# MSE of 2,915
# Strong negative correlation between actual ERP and residuals
# Systematically under-predicting ERP
# Poor predictions for high-performance CPUs

###################################
### Improve model

### Hyperparameter tuning (exhaustive grid search) 

# number of estimators at which boosting is terminated
# experimentation suggests that better performance 
# is found for small number of estimators on this dataset
n_estimators = [2, 4, 6, 8, 10]
# loss function to use when updating weights
loss = ["linear", "square", "exponential"]
# learning rate
learning_rate = list(np.arange(0.1, 2, 0.2))
# Create the grid
grid = {"model__n_estimators": n_estimators,
        "model__loss": loss,
        "model__learning_rate": learning_rate,
        "model__random_state": [42]}

model = AdaBoostRegressor()
# carry out grid search
grid_search(X_train, y_train, model, grid)
# fit and test model with optimal parameter values
ABR_2 = AdaBoostRegressor(random_state=42,
                          n_estimators=4,
                          learning_rate=0.6,
                          loss="linear")

test_model(X_train, y_train, X_test, y_test, ABR_2)
# MSE of 2,714 (better than model 1)
# Strong negative correlation between actual ERP and residuals
# Systematically under-predicting ERP
# Performance being dragged down by one really poor prediction
# vast majority of predictions within +/- 50 units of ERP

### Feature engineering
# Could try discretising the PRP variable (KBinsDiscretizer)