# House-price-predictionAzure
Machine Learning

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from scipy import stats
from sklearn.preprocessing import StandardScaler

import os

%matplotlib inline
warnings.filterwarnings('ignore')

def save_model(model, file_name):
    dir_name = '/kaggle/working/'
    location = os.path.join(dir_name + file_name)
    joblib.dump(model, location)
    
def load_model(file_name):
    dir_name = '/kaggle/working/'
    location = os.path.join(dir_name + file_name)
    return joblib.load(location)

 # Read the CSV
train_csv_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
test_csv_path = '/kaggle/input/house-prices-advanced-regression-techniques//test.csv'
train_set = pd.read_csv(train_csv_path)
test_set = pd.read_csv(test_csv_path)

#  Keep original data clean
train_data = train_set.copy()
test_data = test_set.copy()
test_data.columns

train_data['SalePrice'].describe()

plt.figure(figsize=(16, 6))
sns.distplot(train_data['SalePrice']);

var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', figsize=(16, 6), ylim=(0,800000));

var = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 10))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);


var = 'YearBuilt'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
f, ax = plt.subplots(figsize=(22, 12))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


corrs_matrix = train_data.corr()
f, ax = plt.subplots(figsize=(12, 12))
sns.heatmap(corrs_matrix, vmax=.8, square=True);

sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'OverallCond', 'TotalBsmtSF', 'MSZoning', ]
sns.pairplot(train_data[cols], size = 2.5)
plt.show();

total = train_data.isnull().sum().sort_values(ascending=False)
percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

train_data = train_data.drop((missing_data[missing_data['Total'] > 1]).index,1)
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)

saleprice_scaled = StandardScaler().fit_transform(train_data['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('Low range of distribution:')
print(low_range)
print('\nHigh range of the distribution:')
print(high_range)

var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), figsize=(16, 8));

var = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), figsize=(16, 8));



sns.distplot(train_data['SalePrice'], fit=stats.norm);
fig = plt.figure()
fig.figsize=(16, 38)
res = stats.probplot(train_data['SalePrice'], plot=plt)




