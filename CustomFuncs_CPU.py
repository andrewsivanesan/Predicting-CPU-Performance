"""
Author - Andrew Sivanesan

Custom functions for use in CPU machine learning project
"""
import numpy as np
#import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor, LocalOutlierFactor
from sklearn.svm import SVR
#from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

# define models to compare
def get_models():
    models, names = list(), list()
    models.append(LinearRegression())
    names.append("LR")
    models.append(HuberRegressor())
    names.append("HR")
    models.append(RANSACRegressor(random_state=42))
    names.append("RANSAC")
    models.append(TheilSenRegressor(random_state=42))
    names.append("TSR")
    models.append(Ridge())
    names.append("Ridge")
    models.append(KNeighborsRegressor())
    names.append("KNN")
    models.append(DecisionTreeRegressor())
    names.append("DTR")
    models.append(RandomForestRegressor(random_state=42))
    names.append("RFR")
    models.append(SVR(gamma='auto'))
    names.append("SVR")
    models.append(AdaBoostRegressor(random_state=42))
    names.append("Ada")
    models.append(BaggingRegressor(random_state=42))
    names.append("Bag")
    models.append(GradientBoostingRegressor(random_state=42))
    names.append("GBR")
    return models, names

def create_X_transformer():
   result = ColumnTransformer(transformers=[("YJ", PowerTransformer(method="yeo-johnson"), [0])])
   return result

def train_single_model(X, y, model):
	# define evaluation procedure
	cv = KFold(n_splits=10)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
	return scores

# Note: this function assumes that X_train is one-dimensional
def replace_outliers_1d(X):
    # copy the array
    result = X.copy()
    # calculate interquartile range of the single explanatory variable
    q25, q75 = np.percentile(X, 25), np.percentile(X, 75)
    IQR = q75 - q25
    # calculate the outlier cutoff
    cut_off = IQR * 1.5
    # calculate Tukey's fences
    lower, upper = q25 - cut_off, q75 + cut_off
    # select indexes that are out of bounds
    ix = np.where(np.logical_or(X < lower, X > upper))[0]
    # replace outliers with median value
    result[ix] = np.median(X)
    return result

# def remove_outliers(X):
#     # identify outliers in X
#     lof = LocalOutlierFactor()
#     yhat = lof.fit_predict(X)
#     # filter for all rows in X that are not outliers
#     mask = yhat != -1
#     result = X[mask, :]
#     return result

def fit_test_pipeline(X_train, y_train, model):
    X_transformer = create_X_transformer()
    ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
    result = Pipeline(steps=[("transform", X_transformer), 
                             ("ReplaceOutliers", ReplaceOutliers), 
                             ("model", model)]).fit(X_train, y_train)
    return result

def fit_test_pipeline_KBins(X_train, y_train, model, bins, strat):
    pipe = get_bins_pipeline(model, bins, strat)
    result = pipe.fit(X_train, y_train)
    return result

# no outlier replacement
# def fit_test_pipeline2(X, y, model):
#     X_transformer = create_X_transformer2()
#     result = Pipeline(steps=[("transform", X_transformer), 
#                              ("model", model)]).fit(X, y)
#     return result

# Note: assumes 1D input
# column name hard-coded
def train_multiple_models(X, y):
    models, names = get_models()
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in range(len(models)):
    	# define transformer
        ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
        X_transformer = create_X_transformer()
        # define pipeline
        pipeline = Pipeline(steps=[("transform", X_transformer), 
                                   ("ReplaceOutliers", ReplaceOutliers), 
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

def get_bins_pipeline(model, num_bins, strat):
    ct = ColumnTransformer(transformers=[("Bins", KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy=strat), [0])])
    result = Pipeline(steps=[("transform", ct),  
                             ("model", model)])
    return result

# No outlier replacement
def train_multiple_models_Kbins(X, y, model, strat):
    bins_list = list(range(2, 11))
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in bins_list:
        # define pipeline
        pipeline = get_bins_pipeline(model, i, strat)
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
    
def residual_plot(y_test, y_pred):
    residual = y_pred - y_test
    sns.scatterplot(x=y_test, y=residual)
    plt.title("Residual plot")
    plt.xlabel(y_test.name)
    plt.ylabel("Residual")
    plt.show()
    
def predicted_vs_actual(y_test, y_pred):
    p1 = min(min(y_pred), min(y_test))
    p2 = max(max(y_pred), max(y_test))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.title("Predictions vs actuals")
    plt.xlabel("Actuals")
    plt.ylabel("Predictions")
    plt.show()

def test_model(X_train, y_train, X_test, y_test, model):
    # fit the pipeline to the training set
    pipeline_test = fit_test_pipeline(X_train, y_train, model)
    y_pred = pipeline_test.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    residual_plot(y_test, y_pred)
    predicted_vs_actual(y_test, y_pred)
    
def test_model_KBins(X_train, y_train, X_test, y_test, model, bins, strat):
    # fit the pipeline to the training set
    pipeline_test = fit_test_pipeline_KBins(X_train, y_train, model, bins, strat)
    y_pred = pipeline_test.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    residual_plot(y_test, y_pred)
    predicted_vs_actual(y_test, y_pred)
  
# no outlier replacement
# def test_model2(X_train, y_train, X_test, y_test, model):
#     # fit the pipeline to the training set
#     pipeline_test = fit_test_pipeline2(X_train, y_train, model)
#     y_pred = pipeline_test.predict(X_test)
#     print(mean_squared_error(y_test, y_pred))
#     residual_plot(y_test, y_pred)
#     predicted_vs_actual(y_test, y_pred)
    
# def evaluate_k(X_train, y_train, k_values):
#     # initialise results list
#     results = []
#     for k in k_values:
#         model = KNeighborsRegressor(n_neighbors=k)
#         # define transformer
#         ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
#         X_transformer = create_X_transformer()
#         # define pipeline
#         pipeline = Pipeline(steps=[("transform", X_transformer), 
#                                    ("ReplaceOutliers", ReplaceOutliers), 
#                                    ("model", model)])
#     	# evaluate the model
#         scores = train_single_model(X_train, y_train, pipeline)
#         score = np.median(scores)
#         # store results
#         results.append(scores)
#         print('> k=%d, MSE: %.3f' % (k, score))

# randomised grid search
# 10-fold cross-validation
# negative mean squared error
def grid_search(X, y, model, grid):
    X_transformer = create_X_transformer()
    ReplaceOutliers = FunctionTransformer(func=replace_outliers_1d)
    pipe = Pipeline(steps=[("transform", X_transformer), 
                           ("ReplaceOutliers", ReplaceOutliers),
                           ("model", model)])
    
    # Random search of parameters, using 10-fold cross validation, 
    # search across 10 different combinations
    search = GridSearchCV(estimator=pipe, 
                          param_grid=grid, 
                          cv=KFold(n_splits=10),
                          scoring="neg_mean_squared_error",
                          verbose=0).fit(X, y)
    return search.best_params_, search.best_score_

# # no outlier replacement
# def grid_search2(X, y, model, grid):
#     X_transformer = create_X_transformer2()
#     pipe = Pipeline(steps=[("transform", X_transformer), 
#                            ("model", model)])
    
#     # Random search of parameters, using 10-fold cross validation, 
#     # search across 10 different combinations
#     search = GridSearchCV(estimator=pipe, 
#                           param_distributions=grid, 
#                           cv=KFold(n_splits=10),
#                           scoring="neg_mean_squared_error",
#                           verbose=0, 
#                           random_state=42).fit(X, y)
#     return search.best_params_, search.best_score_