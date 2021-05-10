# Predicting-CPU-Performance

## General notes

* Data for this project is available in text format - **MachineData.txt**
* Exploratory data analysis can be found in **CPUPerformance_EDA.py.** Plots are saved in the EDA plots folder of this repository.
* Main modelling work carried out in the following scripts
*    **PredictingCPUPerformance.py** (helper functions from **CustomFuncs_CPU.py**)
*    **PredictingCPUPerformance_KBinsPRP.py** (helper functions from **CustomFuncs_CPU_KBins.py**)
*    **PredictingCPUPerformance_FeatureEngineering.py** (helper functions from **CustomFuncs_CPU_KBins.py**)
* Directory file paths in the above scripts point to my local C drive. If running the scripts on your own computer please **change the file paths** accordingly.

## Executive summary
The best performing model found in this project is an **AdaBoost regressor** with **six estimators** (decision tree regressors) and a **learning rate of 0.6.** This is applied to one explanatory variable - **published relative performance (PRP)** - with a Yeo-Johnson transformation applied to it to address its heavy right skew, and any outliers replaced with the median (Yeo-Johnson-transformed) PRP value.

The aforementioned model has a mean squared error (MSE) of **2,714** on the test set. This is **49,136 MSE units less than the naive mean benchmark model.** On average the model's predictions are **within 52 units of the actual estimated relative performance (ERP)** of the given CPU.

A single explanatory variable - PRP - was used due to the **strong correlation between all other explanatory variables.**

Feature engineering - namely k-bins discretisation and the inclusion of minimum, median and maximum PRP values by vendor - did not improve model performance beyond that of the model described above.

If additional time were available the following could be explored in pursuit of a more performant model:

* pass **all available explanatory variables** to a decision tree regressor and **identify any additional important features**. The inclusion of more explanatory variables may lead to a performance improvement in the AdaBoost regressor.
* further explore **feature engineering** - particularly if the above yields more explanatory variables to include in the modelling.

## Problem statement

The aim of this project is to predict the **estimated relative performance (ERP)** of central processing units (CPUs) using CPU performance attributes as per the supplied data.

Such a model would reduce the need for physical experiments to verify CPU manufacturers' performance claims and hence reduce experimentation costs (e.g. lab time, technicians).

The data is sourced from https://archive.ics.uci.edu/ml/datasets/Computer+Hardware.

ERP is a continuous variable, so this is a regression problem. Therefore the following models are explored in this project:
    * linear regression
    * k-nearest neighbours regressor
    * support vector regression
    * decision tree regressor
    * random forest regressor
    * Theil-Sen regression
    * Huber regression
    * Ridge regression
    * RANSAC regression
    * AdaBoost regressor
    * Bagging regressor
    * Gradient boosting regressor
    
Performance is measured using **mean squared error**, and benchmarked against the performance of a naive mean prediction model (MSE of **51,850**).
