
import numpy as np
import sys
import os
from dgps import dgp
from scores import mu_hat
from sklearn.metrics.pairwise import rbf_kernel
from scipy.special import logsumexp
import time
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.optimizers import markowitz_eValueOptimizer_returnx
import dacs
import cvxpy as cp

start = time.time()
gammas = [0.075, 0.1, 0.125]
alphas = [0.05, 0.2, 0.35]
variants = []
variant_counter = 0
for gamma_indexer in range(3):
  for job in range(250):
    for setting in range(2):
      for alpha_ind in range(3):
        variants.append((job, setting, gamma_indexer, alpha_ind))
        variant_counter += 1

# use one command-line argument to get the variant_block
variant_block = int(sys.argv[1])

ml_alg_ind = 1
couple = True
for variant in tqdm(np.arange(variant_block*100, (variant_block+1)*100)):
  job, setting, gamma_indexer, alpha_ind = variants[variant]
  gamma = gammas[gamma_indexer]
  np.random.seed([job, setting, alpha_ind])


  noise = 1.

    
  def get_scores(X, Y, mu_hat):
    return np.where(Y > 0, np.inf, -mu_hat.predict(X))


  diversities = []

  train = 1000
  n = 500
  m = 100
  alpha = [0.05, 0.2, 0.35][alpha_ind]
  num_mc_samples_per_step = 25


  trainX, trainY = dgp(train, noise, setting)
  calibX, calibY = dgp(n, noise, setting)
  testX, testY = dgp(m, noise, setting)

  muHat = mu_hat(ml_alg_ind, trainX, trainY)
  calibS = get_scores(calibX, calibY, muHat)
  testS = get_scores(testX, np.zeros(m), muHat)


  combinedX = np.concatenate((calibX, testX))
  similarityMatrix = rbf_kernel(combinedX, combinedX)

  combinedS = np.concatenate((calibS, testS))
  argsortedS = np.argsort(combinedS)

  numCalibBelow = n
  numTestBelow = m

  reached_BH_stop = False

  for indexer in list(reversed(range(n+m))):
    FDP_hat = ((1.+numCalibBelow)/max(1.,numTestBelow))*(m/(n+1.))

    if FDP_hat <= alpha:
      reached_BH_stop = True
    
    if reached_BH_stop:
        eValues = (testS <= combinedS[argsortedS[indexer]]).astype(float) * \
            (n+1.)/(1.+numCalibBelow)
        if numTestBelow > 0:
          if (n+1.)/(1.+numCalibBelow) >= m/(alpha*numTestBelow):
              mask = (eValues > 0).astype(bool)
              numActive = mask.sum()
              x = np.zeros(m)
              
              y = cp.Variable(numActive)
              prob = cp.Problem(cp.Minimize((gamma*0.5*cp.quad_form(y, cp.psd_wrap((similarityMatrix[n:,n:])[mask][:,mask]))) \
                                          - np.ones(numActive).T @ y),
                          [0 <= y,
                          y <= (alpha/m) * (n+1.)/(1.+numCalibBelow) * (np.ones(numActive).T @ y),
                          y <= 1
                          ])
              prob.solve(verbose=False, solver=cp.MOSEK)
              x_solution = y.value
              x[mask] = x_solution

              for b in range(num_mc_samples_per_step):
                  rejections = np.random.binomial(1,p=np.clip(x,0.,1.))
                  
                  diversity = np.sum(rejections) \
                      - gamma*0.5*np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,\
                                                  rejections.astype(bool)])
                  
                  diversities.append(diversity)

      # sample a set randomly

    numCalibBelow -= int(argsortedS[indexer] < n)
    numTestBelow -= int(argsortedS[indexer] >= n)


  with open(f"optimized_markowitz_histogram_results/histogram_v{variant}.csv", "at") as file:
    file.write(",".join(map(str, diversities)) + "\n")

end = time.time()
