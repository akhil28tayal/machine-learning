import numpy as np
import pandas as pd

dataset = pd.read_csv('ex1data1.csv', header=None, dtype=float)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X)

plt.scatter(X, y, marker='x', color='red')
plt.plot(X, y_pred, color='blue')
plt.show()



