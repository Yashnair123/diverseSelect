from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR    
import numpy as np


def mu_hat(ml_alg_ind,trainX, trainY):
    mu_hats = [lin_mu_hat,\
               mlp_mu_hat,\
                svm_mu_hat]
    return mu_hats[ml_alg_ind](trainX,trainY)

def lin_mu_hat(trainX, trainY):
  regr = LinearRegression()
  regr.fit(np.array(trainX), np.array(trainY))
  return regr


def mlp_mu_hat(trainX, trainY):
  regr = MLPRegressor(hidden_layer_sizes=[128,32], max_iter=5000)
  regr.fit(np.array(trainX), np.array(trainY))
  return regr


def svm_mu_hat(trainX, trainY):
  regr = SVR()
  regr.fit(np.array(trainX), np.array(trainY))
  return regr