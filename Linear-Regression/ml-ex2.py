#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:49:12 2017

@author: akhiltayal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def featureScaling(X):
    mu = X.mean()
    sigma = X.std()
    X_norm = (X - mu)/sigma
    return [X_norm, mu, sigma]

def computeCostFunction(X, y, theta):
    m = len(X)
    prediction = np.array([0] * len(X), dtype=float)
    for i in range(0,len(X.columns)):
        prediction = prediction + theta[i] * X[i] 
    sqrError = (prediction - y) ** 2
    J = sqrError.sum() / (2 * m)
    return J

def gradientDescent(X, y, theta, alpha, num_iter):
    m = len(X)
    J = np.array([0]*num_iter, dtype=float)
    for i in range(0, num_iter):
        theta_temp = theta
        prediction = 0
        for j in range(0, len(X.columns)):
            prediction = prediction + theta_temp[j] * X[j]
        for j in range(0, len(X.columns)):
            theta_temp[j] = theta_temp[j] - ((alpha/m) * ((prediction - y) * X[j]).sum())
        theta = theta_temp
        J[i] = computeCostFunction(X, y, theta)
    return [theta, J]
        

if __name__ == "__main__":
    data = pd.read_csv('ex1data2.csv', header=None, dtype=float)
    X = data.iloc[:,:-1]
    y = data.iloc[:, -1]
    [X, mu, sigma] = featureScaling(X)
    [y, mu_y, sigma_y] = featureScaling(y)
    temp = pd.DataFrame([1] * len(X))
    X = pd.concat([temp, X], axis=1)
    X.columns = [0, 1, 2]
    
    theta = np.array([0] * len(X.columns), dtype=float)
    J_zero = computeCostFunction(X, y, theta)
    
    [theta, J] = gradientDescent(X, y, theta, alpha=0.01, num_iter=1500)
    
    y_pred = np.array([0] * len(X), dtype=float)
    for i in range(0, len(X.columns)):
        y_pred = y_pred + theta[i] * X[i]