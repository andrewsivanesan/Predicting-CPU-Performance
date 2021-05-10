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
from sklearn.metrics import mean_squared_error

wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
from CustomFuncs_CPU import get_models, train_single_model, residual_plot, predicted_vs_actual

# def get_bins_pipeline(model, num_bins, strat):
#     ct = ColumnTransformer(transformers=[("Bins", KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy=strat), [0])])
#     result = Pipeline(steps=[("transform", ct),  
#                              ("model", model)])
#     return result

def fit_test_pipeline_KBins(X_train, y_train, model, ct):
    pipe = Pipeline(steps=[("transform", ct),  
                           ("model", model)])
    result = pipe.fit(X_train, y_train)
    return result

def train_multiple_models_KBins(X, y, model, strat):
    bins_list = list(range(2, 11))
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in bins_list:
        ct = ColumnTransformer(transformers=[("Bins", KBinsDiscretizer(n_bins=i, encode='ordinal', strategy=strat), [0])])
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
    
def test_model_KBins(X_train, y_train, X_test, y_test, model, ct):
    # fit the pipeline to the training set
    pipeline_test = fit_test_pipeline_KBins(X_train, y_train, model, ct)
    y_pred = pipeline_test.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    residual_plot(y_test, y_pred)
    predicted_vs_actual(y_test, y_pred)
    
def train_multiple_models_2(X, y, ct):
    models, names = get_models()
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in range(len(models)):
        # define pipeline
        pipeline = Pipeline(steps=[("transform", ct), 
                                   ("model", models[i])])
    	# evaluate the model
        scores = train_single_model(X, y, pipeline)
        # store results
        results.append(scores)
    	# print summary statistics of results
        print('>%s %.3f (%.3f)' % (names[i], np.median(scores), np.std(scores)))
    
    # Visually compare algorithms using box plots
    plt.boxplot(results, labels=names)
    plt.title("Model comparison")
    plt.ylabel("Neg MSE")
    plt.show() 
  
# randomised grid search
# 10-fold cross-validation
# negative mean squared error
# def grid_search(X, y, model, grid):
#     X_transformer = create_X_transformer()
#     ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
#     pipe = Pipeline(steps=[("transform", X_transformer), 
#                            ("ReplaceOutliers", ReplaceOutliers),
#                            ("model", model)])
    
#     # Random search of parameters, using 10-fold cross validation, 
#     # search across 10 different combinations
#     search = GridSearchCV(estimator=pipe, 
#                           param_grid=grid, 
#                           cv=KFold(n_splits=10),
#                           scoring="neg_mean_squared_error",
#                           verbose=0).fit(X, y)
#     return search.best_params_, search.best_score_