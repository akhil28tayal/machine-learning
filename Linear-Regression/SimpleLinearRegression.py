#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:23:12 2017

@author: akhiltayal
"""

# Simple Linear Regression

# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def importData(filename):
    dataset = pd.read_csv(filename, header=None, dtype=float)
    X = dataset.iloc[:, 0].values
    y = dataset.iloc[:, 1].values
    return [X, y]

# Cost Functoin J(theta) = (1/2*m) * sum(h(x) - y)^2
# h(x) = theta[0] + theta[1] * x (Prediction)
def computeCost(X, y, theta):
    m = len(X)
    prediction = theta[0] + (theta[1] * X)
    sqrError = (prediction - y) ** 2
    J = sqrError.sum() / (2*m)
    return J

# Gradient Descent theta[j] = theta[j] - derivative(J(theta))
# theta[j] = theta[j] - (sum(h(theta) - y) * x[j] ) * (alpha/m)
# x[0] = 1
def gradientDescent(X, y, theta, alpha, num_iter):
    m = len(X)
    J = np.array([0] * num_iter, dtype=float)
    for i in range(0, num_iter):
        prediction = theta[0] + (theta[1] * X)
        theta_zero = theta[0] - ((prediction - y).sum())*(alpha/m)
        theta_one = theta[1] - (((prediction - y) * X).sum())*(alpha/m)
        theta = [theta_zero, theta_one]
        J[i] = computeCost(X, y, theta)
    return [theta, J]

# Graphical Visualisation of the learning algorithm
def visualisingData(X, y, theta, J):
    y_pred = theta[0] + theta[1] * X
    plt.scatter(X, y, marker='x', color='red')
    plt.plot(X, y_pred, color='blue')
    plt.show()

    plt.plot(J)
    plt.show()

# Main Function
if __name__ == "__main__":

    [X, y] = importData("ex1data1.csv")

    theta = np.array([0, 0], dtype=float)

    J_zero = computeCost(X, y, theta)

    [theta, J] = gradientDescent(X, y, theta, alpha=0.01, num_iter=1500)

    visualisingData(X, y, theta, J)
