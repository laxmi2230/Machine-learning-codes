# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 12:06:33 2021

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = ColumnTransformer(
      transformers=[
        ("OneHot",        # Just a name
         OneHotEncoder(), # The transformer class
         [3]              # The column(s) to be applied on.
         )
    ],
    remainder='passthrough' # donot apply anything to the remaining columns
)
X = onehotencoder.fit_transform(X.tolist())
X = X.astype('float64')

#to avoid california
X = X[:, 1:]

#splitting dataset to training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


#building optimal model using backward elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis=1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [0,4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())
X_opt = X[:, [4,5]]
regressor_OLS = sm.OLS(endog = y,exog = X_opt).fit()
print(regressor_OLS.summary())