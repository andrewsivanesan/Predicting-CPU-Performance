"""
Author - Andrew Sivanesan

This script explores the CPU dataset
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###########################
# Exploratory data analysis
###########################

wd = "C:\\Users\\apsiv\\Documents\\GitHub\\Predicting-CPU-Performance"
os.chdir(wd)
filename = "MachineData.txt"
cols = ["vendor", "model", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
# read in data
df = pd.read_csv(wd + "\\" + filename, sep=",", names=cols)

# check for missing values
df.isnull().any()
# no missing values in dataset

# get summary statistics
df_summary = df.describe()
# will need to standardise continuous variables due to 
# differing magnitudes and units of measurement

dups = df.duplicated()
# no duplicates

#########################
### Continuous variables
##########################

discrete_cols = df.select_dtypes(include=['object', 'bool']).columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# plot separate boxplots for each continuous variable
plot_wd = r"C:\Users\apsiv\Documents\GitHub\Predicting-CPU-Performance\EDA plots"
os.chdir(plot_wd)

for i in numeric_cols:
    sns.boxplot(data=df, x=i)
    plt.savefig(i + "_BoxPlot.png")
    # clear current figure
    plt.clf()
    
# All the continuous variables are right skewed and have outliers
# plot scatter plots of estimated relative performance against
# each of the continuous explanatory variables
for i in numeric_cols:
    if i != "ERP":
        sns.scatterplot(data=df, x=i, y="ERP")
        plt.savefig(i + "_ScatterPlot.png")
        # clear current figure
        plt.clf()
# heteroskedasticity in most of the continuous explanatory variables
# exponential decay between MYCT and ERP
# exponential relationship between MMAX and ERP
# strong positive linear relationship between PRP and ERP
    
#########################
### Discrete variables
##########################
df["vendor"].value_counts()
# IBM is over-represented in the dataset. Vendor classes are imbalanced.
df["model"].value_counts()
# models are uniformly represented (each observation relates to a distinct model)

########################
### Correlation
#######################

# correlation heatmap
corr = df.corr(method="spearman")
sns.heatmap(corr, cmap="viridis")
plt.savefig("CorrelationHeatMap.png")

# strong correlations between:
#   - MYCT and MMIN, MMAX, CACH, CHMIN, CHMAX, PRP, ERP
#   - MMIN and MMAX, CACH, CHMIN, PRP
#   - MMAX and CACH, PRP, ERP
#   - CACH and PRP, ERP
#   - CHMIN and CHMAX, PRP, ERP
#   - CHMAX and PRP, ERP
#   - PRP and ERP

# We could therefore begin by using PRP as the sole predictor in our model