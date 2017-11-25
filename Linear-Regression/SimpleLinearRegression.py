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

def computeCost(X, y, theta):
    m = len(X)
    prediction = theta[0] + (theta[1] * X)
    sqrError = (prediction - y) ** 2
    J = sqrError.sum() / (2*m)
    return J

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

def visualisingData(X, y, theta, J):
    y_pred = theta[0] + theta[1] * X
    plt.scatter(X, y, marker='x', color='red')
    plt.plot(X, y_pred, color='blue')
    plt.show()
    
    plt.plot(J)
    plt.show()
    
if __name__ == "__main__":
    
    [X, y] = importData("ex1data1.csv")
    
    theta = np.array([0, 0], dtype=float)
    
    J_zero = computeCost(X, y, theta)
    
    [theta, J] = gradientDescent(X, y, theta, alpha=0.01, num_iter=1500)
    
    visualisingData(X, y, theta, J)