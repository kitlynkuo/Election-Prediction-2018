# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:47:26 2018

@author: Marco
"""

import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler

""" import data """
data = pd.read_excel('X_non_district_senate.xlsx', na_values = 'NA')
data1 = data.dropna(axis = 0)

y = data1.iloc[:, -3::]
x = data1.iloc[:, 2:-3]



""" merge columns """
x['Age - 18 to 34'] = x['Total population - AGE - 20 to 24 years'] +\
                         x['Total; Estimate; Total population - AGE - 25 to 29 years'] +\
                         x['Total; Estimate; Total population - AGE - 30 to 34 years']

x['Age - 35 to 49'] = x['Total; Estimate; Total population - AGE - 35 to 39 years'] +\
                         x['Total; Estimate; Total population - AGE - 40 to 44 years'] +\
                         x['Total; Estimate; Total population - AGE - 45 to 49 years']
                         
x['Age - over 50'] = x['Total; Estimate; Total population - AGE - 50 to 54 years'] +\
                         x['Total; Estimate; Total population - AGE - 55 to 59 years'] +\
                         x['Total; Estimate; Total population - AGE - 60 to 64 years'] +\
                         x['Total; AGE voer 65']

x['Income - 10k to 35k'] = x['Households; Estimate; Less than $10,000'] +\
                         x['Households; Estimate; $10,000 to $14,999'] +\
                         x['Households; Estimate; $15,000 to $24,999'] +\
                         x['Households; Estimate; $25,000 to $34,999']

x['Income - 35k to 100k'] = x['Households; Estimate; $35,000 to $49,999'] +\
                         x['Households; Estimate; $50,000 to $74,999'] +\
                         x['Households; Estimate; $75,000 to $99,999']
                         
x['Income - over 100k'] = x['Households; Estimate; $100,000 to $149,999'] +\
                         x['Households; Estimate; $150,000 to $199,999'] +\
                         x['Households; Estimate; $200,000 or more'] 

x['Race - White'] = x['Population of one race: - White alone']
x['Race - Black'] = x['Population of one race: - Black or African American alone']
x['Race - Asian'] = x['Population of one race: - Asian alone']
x['Race - Others'] = x['Population of one race: - American Indian and Alaska Native alone'] +\
                     x['Population of one race: - Native Hawaiian and Other Pacific Islander alone']

x['Education - Below High School'] = x['Estimate; Less than high school graduate'] +\
                                     x['Estimate; High school graduate (includes equivalency)']
x['Education - Above College'] = x['Estimate; Some college or associate degree'] +\
                                 x['Estimate; Bachelor degree']
x['Education - Above Graduate'] = x["Estimate; Graduate or professional degree"]

#x['Gender - Male to Female Ratio'] = x['Tot_Pop_Male_over18'] / x['Tot_Pop_Female_over18']
            
# select desired attributes             
x_sim = x.iloc[:, 54:77]
x_sim.index = np.arange(0, 11)


#writer = pd.ExcelWriter('X.xlsx')
#x_sim.to_excel(writer, 'Sheet1')
#writer.save()


""" normalization """ 
# separate categorical and numeric data
x_numeric = x_sim.iloc[:, 4::]
scaler = StandardScaler().fit(x_numeric)
x_normalize = scaler.transform(x_numeric)

#x_normalize = pd.concat([x_sim.iloc[:, 0:4], pd.DataFrame(x_normalize)], axis=1)



""" PCA """
from sklearn.decomposition import PCA
pca = PCA(n_components = 11)
x_pca = pd.DataFrame(pca.fit_transform(x_normalize))

# combine categorical and numeric
x_pca = pd.concat([x_sim.iloc[:, 0:4], x_pca], axis=1)



""" Train Test split """
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.1, random_state=None)



""" MultiOutputRegressor """
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

""" Grid Search """
from sklearn.model_selection import GridSearchCV
GCV = 5

""" cross val score """
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


""" Lasso regression """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__alpha': [1, 1e2, 1e3, 1e4, 1e5]}
lasso_reg = MultiOutputRegressor(linear_model.Lasso())
grid = GridSearchCV(lasso_reg, parameters, cv=GCV, scoring='neg_mean_absolute_error')
grid.fit(x_train, y_train)
grid.best_params_ 
grid.best_score_  

# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))



""" Ridge regression """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__alpha': [1, 1e1, 1e2, 1e3, 1e4, 1e5], 'estimator__solver': ['svd', 'cholesky', 'sag']}
ridge_reg = MultiOutputRegressor(Ridge())
#parameters = {'alpha': [1, 1e1, 1e2, 1e3, 1e4, 1e5], 'solver': ['svd', 'cholesky', 'sag']}
#ridge_reg = Ridge()
grid = GridSearchCV(ridge_reg, parameters, cv=GCV, scoring='neg_mean_absolute_error')
grid.fit(x_train, y_train)
grid.best_params_ 
grid.best_estimator_
#grid.cv_results_ 

# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))



""" SVM regression """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__kernel': ['linear', 'rbf'], 'estimator__C': [1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7],\
              'estimator__epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]}
svm_reg = MultiOutputRegressor(SVR())
grid = GridSearchCV(svm_reg, parameters, cv=GCV, scoring = 'neg_mean_absolute_error')
grid.fit(x_train, y_train)
grid.best_params_ 


# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))



""" Bayesian Ridge regression """""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__alpha_1': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 'estimator__alpha_2': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],\
              'estimator__lambda_1': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8], 'estimator__lambda_2': [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]}
bay_reg = MultiOutputRegressor(linear_model.BayesianRidge())
grid = GridSearchCV(bay_reg, parameters, cv=GCV, scoring = 'neg_mean_absolute_error')
grid.fit(x_train, y_train)
grid.best_params_ 


# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))



""" RandomForest """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__bootstrap': ['True', 'False'], 'estimator__max_features': ['auto', 'sqrt', 'log2'],\
              'estimator__min_samples_split': [2, 3, 4, 5, 6], 'estimator__min_samples_leaf': [1, 2, 3, 4, 5]}
ranforest = MultiOutputRegressor(RandomForestRegressor())
grid = GridSearchCV(ranforest, parameters, cv=GCV, scoring = 'neg_mean_absolute_error')
grid.fit(x_train, y_train)
grid.best_params_

 
# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))



""" GradientBoosting """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
parameters = {'estimator__n_estimators': [25, 50, 100, 200], 'estimator__loss': ['ls', 'lad'],\
              'estimator__max_depth': [2, 3, 4, 5, 6], 'estimator__learning_rate': [1e-4, 1e-3, 1e-2, 1e-1, 1],\
              'estimator__subsample': [0.2, 0.4, 0.6, 0.8, 1.0]}
GraBoost = MultiOutputRegressor(GradientBoostingRegressor())
grid = GridSearchCV(GraBoost, parameters, cv=GCV, scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
grid.best_params_ 


# Voter output: True vs Prediction
print("y true: {0} \ny_pred: {1} ".format(np.array(y_test), np.array((np.round(grid.predict(x_test))))))
print("training score: {0} \ntest score: {1} ".format(grid.best_score_, grid.score(x_test, y_test)))

# mean absolute error
mean_absolute_error(y_train, grid.predict(x_train), multioutput = 'uniform_average')
mean_absolute_error(y_test, grid.predict(x_test), multioutput = 'uniform_average')

# cross validation
score = cross_val_score(grid.best_estimator_, x_normalize, y, cv=5, scoring = 'neg_mean_absolute_error')
print("(mean, std) = ({0}, {1})".format(np.mean(score).round(), np.std(score).round()))