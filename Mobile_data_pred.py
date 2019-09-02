# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:29:18 2019

@author: vkuma140
"""

# packages
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data=pd.read_csv('C:\\Users\\vkuma140\\Music\\Python\\Mobile data  prediction\\train.csv')
print(data)

data.info()
data.describe()
data.isnull().sum()

# =============================================================================
#  Data Analysis
#  Visualizations
#  Categorical Feature
# =============================================================================


categorical = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']


plt.figure(figsize=(15, 8))
# set width of bar
bar_width = 0.1

# set height of bar
bars1 = [len(data[data[col] == 1]) for col in data[categorical]]
bars2 = [len(data[data[col] == 0]) for col in data[categorical]]

# set position of bar on X axis

r1 = np.arange(len(bars1))
r2 = [x + bar_width for x in r1]

# make the plot
plt.bar(r1, bars1, color='#6ef472', width=bar_width, edgecolor='white', label='True')
plt.bar(r2, bars2, color='#00bfff', width=bar_width, edgecolor='white', label='False')
        
        
# create legend & show graphic
plt.legend(fontsize=13)
plt.show()


#Categorical data
categorical.append('price_range')
categorical
#Numerical Features
numerical = data.columns.difference(categorical).tolist()

plt.figure(figsize=(8, 4))
sns.pointplot(x='price_range', y='battery_power', data=data)
plt.show()

plt.figure(figsize=(12, 4))
sns.catplot(x='price_range', y='px_height', data=data)
plt.show()

plt.figure(figsize=(12, 4))
sns.catplot(x='price_range', y='battery_power',kind='boxen', data=data)
plt.show()


plt.figure(figsize=(8, 4))
sns.pointplot(x='price_range', y='ram', data=data)
plt.show()



f, ax = plt.subplots(figsize=(15, 8))

# correlation matrix
corr = data.corr()

sns.heatmap(corr)


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#Training the data
X = data.drop('price_range', axis=1)
Y = data.price_range

train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.3,random_state=20)

print('Shape of x_train: ', train_x.shape)
print('Shape of y_train: ', train_y.shape)
print('Shape of x_test: ', test_x.shape)
print('Shape of y_test: ', test_y.shape)


lr=LinearRegression().fit(train_x,train_y)

lr.score(test_x,test_y)#0.909630156215128
lr.score(train_x,train_y)#0.9213005025555815

# #$$$$$$$$$$$$$$$$$$$$$$$$$$$=====================================
# K-Neighbors Classifier
# Hyperparameter Tuning
# =============================================================================


knn1 = KNeighborsClassifier(n_neighbors=5).fit(train_x, train_y)

y_pred=knn1.predict(test_x)

# using confusion matrix for checkin accuracy

Conf_matrix=confusion_matrix(test_y,y_pred)
Conf_matrix
#array([[129,   4,   0,   0],
#       [  5, 141,   5,   0],
#       [  0,   8, 144,  11],
#       [  0,   0,  10, 143]], dtype=int64)



accuracy=accuracy_score(test_y,y_pred)
accuracy#0.9283333333333333

# =============================================================================
# Logistic Regression
# Hyperparameter Tuning
# =============================================================================

clf = LogisticRegression()
param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}

gridsearch = GridSearchCV(clf, param_grid).fit(train_x,train_y)

#GridSearchCV(cv='warn', error_score='raise-deprecating',
#       estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#          intercept_scaling=1, max_iter=100, multi_class='warn',
#          n_jobs=None, penalty='l2', random_state=None, solver='warn',
#          tol=0.0001, verbose=0, warm_start=False),
#       fit_params=None, iid='warn', n_jobs=None,
#       param_grid={'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']},
#       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
#       scoring=None, verbose=0)

gridsearch.best_params_
#{'C': 100, 'penalty': 'l1'}


#%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# USING RANDOM FOREST
# number of trees in random forest
n_estimators = range(1 , 10, 1)

# maximum number of levels in tree
max_depth = range(1 , 10, 1)


# minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
}




# use the random grid to search for best hyperparameters
# first create the base model to tune
rf = RandomForestClassifier()

# random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, 
                               param_distributions=random_grid, 
                               n_iter=100, 
                               cv=3, 
                               verbose=0, 
                               random_state=42 
                             )

# fit the random search model
rf_random.fit(train_x, train_y)

rf_random.best_params_ #
# =============================================================================
# {'n_estimators': 9,
#  'min_samples_split': 10,
#  'min_samples_leaf': 1,
#  'max_depth': 7}
# =============================================================================

rf2=RandomForestClassifier(**rf_random.best_params_).fit(train_x,train_y)

y1_pred=rf2.predict(test_x)

accu=accuracy_score(test_y,y1_pred)#0.795
accu


#Conclusion is that KNN is giving the best accuracy as comparsion to another algorithm that 
#mean we have less error.

import os
os.chdir('C:\\Users\\vkuma140\\Music\\SQL\\ASSIGNMENT')



import pandas as pd
df=pd.read_csv('cars_cluss',index_col="id")
print (df)
print( df[df.columns].notnull()*1 )

matched = len(