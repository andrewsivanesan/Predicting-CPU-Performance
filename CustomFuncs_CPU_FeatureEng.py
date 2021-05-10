"""
Author - Andrew Sivanesan

Custom functions for use in CPU machine learning project
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

# load custom module
wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU_KBins import train_single_model

def train_multiple_models_KBins2(X, y, model, strat):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    bins_list = list(range(2, 11))
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in bins_list:
        ct = ColumnTransformer(transformers=[("Bins", KBinsDiscretizer(n_bins=i, encode='ordinal', strategy=strat), numeric_cols)])
        # define pipeline
        pipeline = Pipeline(steps=[("transform", ct),  
                                   ("model", model)])
    	# evaluate the model
        scores = train_single_model(X, y, pipeline)
        # store results
        results.append(scores)
    	# print summary statistics of results
        print('>%s %.3f (%.3f)' % (i, np.median(scores), np.std(scores)))
        
    # Visually compare algorithms using box plots
    plt.boxplot(results, labels=bins_list)
    plt.title("Bins comparison - " + strat)
    plt.ylabel("Neg MSE")
    plt.show()
    
def remove_outliers(X):
    # identify outliers in X
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    # filter for all rows in X that are not outliers
    #mask = yhat != -1
    mask = [i for i, x in enumerate(yhat) if x == 1]
    result = X.iloc[mask]
    return result
