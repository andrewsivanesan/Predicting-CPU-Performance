# -*- coding: utf-8 -*-
"""
Author - Andrew Sivanesan

Custom function for use in CPU machine learning project
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

# define models to test
def get_models():
    models, names = list(), list()
    models.append(LinearRegression())
    names.append("LR")
    models.append(HuberRegressor())
    names.append("HR")
    models.append(RANSACRegressor())
    names.append("RANSAC")
    models.append(TheilSenRegressor())
    names.append("TSR")
    models.append(KNeighborsRegressor())
    names.append("KNN")
    models.append(DecisionTreeRegressor())
    names.append("DTR")
    models.append(RandomForestRegressor(random_state=42))
    names.append("RFR")
    models.append(SVR(gamma='auto'))
    names.append("SVR")
    
    return models, names

def train_single_model(X, y, model):
	# define evaluation procedure
	cv = KFold(n_splits=10, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
	return scores

def train_multiple_models(X_train, y_train):
    # identify continous variables
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    models, names = get_models()
    # initialise results list
    results = []
    # evaluate each model in turn
    for i in range(len(models)):
    	# define transformer
        X_transformer = ColumnTransformer(transformers=[("BoxCox", PowerTransformer(method="boxcox"), numeric_cols)])
                                                        
        # define pipeline
        pipeline = Pipeline(steps=[("transform", X_transformer), ("model", models[i])])
    	# evaluate the model
        scores = train_single_model(X_train, y_train, pipeline)
        # store results
        results.append(scores)
    	# print summary statistics of results
        print('>%s %.3f (%.3f)' % (names[i], np.median(scores), np.std(scores)))
    
    # Visually compare algorithms using box plots
    plt.boxplot(results, labels=names)
    plt.title("Model comparison")
    plt.ylabel("Neg MSE")
    plt.show()

def test_model(model, X_train, y_train, X_test, y_test):
    # identify continous variables
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
    m = model
    # define transformer
    X_transformer = ColumnTransformer(transformers=[("BoxCox", PowerTransformer(method="boxcox"), numeric_cols)])

    pipeline_test = make_pipeline(X_transformer, m).fit(X_train, y_train)
    y_pred = pipeline_test.predict(X_test)
    print(mean_squared_error(y_test, y_pred))
    
def outlier_flag(x, lower, upper):
    if (x < lower) or (x > upper):
        result = 1
    else:
        result = 0
    return result

# Note: this function assumes that X_train is one-dimensional
def remove_outliers_1d(X_train, y_train):
    # calculate interquartile range of the single explanatory variable
    q25, q75 = np.percentile(X_train, 25), np.percentile(X_train, 75)
    IQR = q75 - q25
    # calculate the outlier cutoff
    cut_off = IQR * 1.5
    # calculate Tukey's fences
    lower, upper = q25 - cut_off, q75 + cut_off
    # create temporary dataframe
    df_tmp = pd.DataFrame(data={"PRP": X_train, "ERP": y_train})
    # identify outlier observations
    df_tmp["outlier"] = df_tmp.apply(lambda r: outlier_flag(r["x"], lower, upper), axis=1)
    # filter for observations that are not outliers
    df_tmp2 = df_tmp[df_tmp["outlier"] == 0]
    # select x and y columns
    X_train_out, y_train_out = df_tmp2["x"], df_tmp2["y"]
        
    return X_train_out, y_train_out