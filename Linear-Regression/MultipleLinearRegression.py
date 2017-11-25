#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:23:12 2017

@author: akhiltayal
"""

# Multiple Linear Regression

# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def importData(filename):
    data = pd.read_csv(filename, header=None, dtype=float)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return [X, y]

# Feature Scaling for better convergence of the Gradient Descent
# X = (X - mean)/std
def featureScaling(X):
    mu = X.mean(0)
    sigma = X.std(0)
    X_norm = (X - mu)/sigma
    return [X_norm, mu, sigma]

# Cost Functoin J(theta) = (1/2*m) * sum(h(x) - y)^2
# h(x) = theta[0] + theta[1] * x (Prediction)
def computeCost(X, y, theta):
    m = len(X)
    prediction = (theta * X).sum(1)
    sqrError = (prediction - y) ** 2
    J = (sqrError)/(2*m)
    return J

# Gradient Descent theta[j] = theta[j] - derivative(J(theta))
# theta[j] = theta[j] - (sum(h(theta) - y) * x[j] ) * (alpha/m)
# x[0] = 1
def gradientDescent(X, y, theta, alpha, num_iter):
    m = len(X)
    for i in range(0, num_iter):
        prediction = (theta * X).sum(1)
        theta_temp = []
        for j in range(0, len(theta)):
            t = theta[j] - (((prediction - y) * X[:, j]).sum())*(alpha/m)
            theta_temp.append(t)
        theta = theta_temp
    return theta

# Graphical Visualisation of the learning algorithm
def visualisingData(X, y, theta):
    y_pred = (theta * X).sum(1)
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    axes.scatter(X[:, 1], X[:, 2], y)
    axes.plot(X[:, 1],  X[:, 2], y_pred)
    plt.show()

# Main Function
if __name__ == "__main__":
    [X, y] = importData('ex1data2.csv')

    [X, mu, sigma] = featureScaling(X)
    [y, mu_y, sigma_y] = featureScaling(y)

    # Adding vector of 1's for X[0]
    one_array = np.array([1]*len(X))
    X = np.column_stack((one_array, X))

    theta = np.array([0] * X.shape[1], dtype = float)

    theta = gradientDescent(X, y, theta, alpha= 0.01, num_iter = 1500)

    visualisingData(X, y, theta)
