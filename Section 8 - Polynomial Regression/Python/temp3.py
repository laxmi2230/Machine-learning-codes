# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 12:17:40 2021

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#for linear regression
from sklearn.linear_model import LinearRegression
line_reg = LinearRegression()
line_reg.fit(X, y)

#for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
line_reg_2 = LinearRegression()
line_reg_2.fit(X_poly, y)

#making graph for linear model
plt.scatter(X, y, color = 'red')
plt.plot(X, line_reg.predict(X), color = 'blue')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#making graph for ploynomial Regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, line_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'green')
plt.title('Truth of Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


