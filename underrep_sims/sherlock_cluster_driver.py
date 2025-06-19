import numpy as np
import sys
import os
from sklearn.linear_model import LinearRegression
from dgps import dgp
from scores import mu_hat
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import cluster_exact_diverseSelect

variants = []
for ml_alg_ind in range(3):
  for setting in range(6):
    for alpha_ind in range(3):
      variants.append((ml_alg_ind, setting, alpha_ind))

# use one command-line argument to get the variant
variant = int(sys.argv[1])
ml_alg_ind, setting, alpha_ind = variants[variant]

np.random.seed([variant])

alphas = [0.05, 0.2, 0.35]

def get_scores(X, Y, mu_hat):
  return np.where(Y > 0, np.inf, -mu_hat.predict(X))

def oracle_mu(X,setting):
  if setting == 0:
    return X
  if setting == 1:
    return (X**2)-1.
  if setting == 2:
    return 2*np.cos(X)#np.cos(X)
  if setting == 3:
    return 1.5*((X > 0).astype(int))-0.75+0.001*np.random.normal(size=X.shape[0])#2*((X > 0).astype(int))-1
  if setting == 4:
    return (X**2)-1.
  if setting == 5:
    return X**3+X


diversities = []
vanilla_diversities = []

freqs = np.zeros(4)
non_zero = 0.
vanilla_freqs = np.zeros(4)
vanilla_non_zero = 0.
cpi0s = []
pi0s = 0
fdps = []
numRs = []

for job in range(250):
  x_noise = 1.
  alpha = alphas[alpha_ind]

  train = 1000
  n = 500
  m = 300
  skip=1


  trainX, trainY, trainComp, components = dgp(train, x_noise, setting)
  calibX, calibY, calibComp, _ = dgp(n, x_noise, setting)
  testX, testY, testComp, _ = dgp(m, x_noise, setting)

  if ml_alg_ind < 3:
    muHat = mu_hat(ml_alg_ind, trainX[:,np.newaxis], trainY)
    calibS = get_scores(calibX[:,np.newaxis], calibY, muHat)
    testS = get_scores(testX[:,np.newaxis], np.zeros(m), muHat)
  else:
    calibS = np.where(calibY > 0, np.inf, -oracle_mu(calibX,setting))
    testS = -oracle_mu(testX,setting)

  vanilla_rejections, _, __ = bh(calibS, testS, n, m, alpha)
  numComponents = components
  componentPredictions = np.concatenate((calibComp, testComp))
  
  start = time.time()
  rejections = cluster_exact_diverseSelect(calibS, testS, n, m, alpha, componentPredictions, \
                                            numComponents, skip)
  end = time.time()

  counts = np.bincount(testComp[rejections.astype(bool)], \
                          minlength=numComponents)
  

  vanilla_counts = np.bincount(testComp[vanilla_rejections.astype(bool)],\
                          minlength=numComponents)
  
  fdp = np.sum([int(testY[i] <= 0)*int(rejections[i] == 1.) \
                        for i in range(m)])/max(1., np.sum(rejections))
  tdp = np.sum([int(testY[i] > 0)*int(rejections[i] == 1.) \
                  for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                  for i in range(m)]))
  num_rejections = np.sum(rejections)

  pi0 = float(np.mean((trainY <= 0).astype(int)))
  component_conditional_pi0s = []
  for c_ind in range(numComponents):
     mask = trainComp == c_ind
     component_conditional_pi0s.append(float(np.mean((trainY[mask] <= 0).astype(int))))

  metrics = [fdp, tdp, num_rejections, end-start]


  vanilla_fdp = np.sum([int(testY[i] <= 0)*int(vanilla_rejections[i] == 1.) \
                        for i in range(m)])/max(1., np.sum(vanilla_rejections))
  vanilla_tdp = np.sum([int(testY[i] > 0)*int(vanilla_rejections[i] == 1.) \
                  for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                  for i in range(m)]))
  vanilla_num_rejections = np.sum(vanilla_rejections)

  vanilla_metrics = [vanilla_fdp, vanilla_tdp, vanilla_num_rejections]

  pi_0arr = [pi0]


  with open(f"results/counts_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, counts)) + "\n")

  with open(f"results/vanilla_counts_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, vanilla_counts)) + "\n")

  with open(f"results/metrics_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, metrics)) + "\n")

  with open(f"results/vanilla_metrics_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, vanilla_metrics)) + "\n")

  with open(f"results/pi0s_v{variant}.csv", "at") as file:
    file.write(",".join(map(str, pi_0arr)) + "\n")
  
  with open(f"results/conditional_pi0s_v{variant}.csv", "at") as file:
    file.write(",".join(map(str, component_conditional_pi0s)) + "\n")

  if job == 0:
    cpi0s = np.zeros_like(counts)
    freqs = np.zeros_like(counts)
    vanilla_freqs = np.zeros_like(vanilla_counts)

  freqs = freqs+counts/np.sum(counts) if np.sum(counts) > 0 else freqs
  vanilla_freqs = vanilla_freqs+vanilla_counts/np.sum(vanilla_counts) if np.sum(vanilla_counts) > 0 else vanilla_freqs
  non_zero += 1 if np.sum(counts) > 0 else 0
  vanilla_non_zero += 1 if np.sum(vanilla_counts) > 0 else 0.

  cpi0s = cpi0s+np.array(component_conditional_pi0s)/250
  pi0s = pi0s + pi0/250

print(freqs/non_zero)
print(vanilla_freqs/vanilla_non_zero)
print(f'cond_pi0s: {cpi0s}')
print(f'pi0: {pi0s}')