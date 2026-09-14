from sklearn.linear_model import LinearRegression   
import numpy as np


def mu_hat(trainX, trainY):
    regr = LinearRegression()
    regr.fit(np.array(trainX), np.array(trainY))
    return regr