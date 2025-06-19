import numpy as np
import sys
import os
from dgps import dgp
from scores import mu_hat
from sklearn.metrics.pairwise import rbf_kernel
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import markowitz_approx_diverseSelect

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

# use one command-line argument to get the variant
couple = False
custom = False

variant_block = int(sys.argv[1])

for variant in np.arange(variant_block*2, (variant_block+1)*2):
  ml_alg_ind = 1
  job, setting, gamma_indexer, alpha_ind = variants[variant]
  print(job, setting, gamma_indexer, alpha_ind)

  gamma = gammas[gamma_indexer]
  np.random.seed([job, setting, alpha_ind])


  noise = 1.

    
  def get_scores(X, Y, mu_hat):
    return np.where(Y > 0, np.inf, -mu_hat.predict(X))


  train = 1000
  n = 500
  m = 100

  alpha = alphas[alpha_ind]
  skip=50
  num_mc_samples = 300


  trainX, trainY = dgp(train, noise, setting)
  calibX, calibY = dgp(n, noise, setting)
  testX, testY = dgp(m, noise, setting)

  muHat = mu_hat(ml_alg_ind, trainX, trainY)
  calibS = get_scores(calibX, calibY, muHat)
  testS = get_scores(testX, np.zeros(m), muHat)


  combinedX = np.concatenate((calibX, testX))
  similarityMatrix = rbf_kernel(combinedX, combinedX)
  print(gamma)
  
  start = time.time()
  rejections, block_indexer, indexer, across_mc_total_time_solving\
                 = markowitz_approx_diverseSelect(calibS, testS, n, m, alpha, gamma, \
                                      similarityMatrix, num_mc_samples, couple, skip, custom)
  end = time.time()

  total_time = end-start


  vanilla_rejections, vanilla_indexer, __ = bh(calibS, testS, n, m, alpha)

  diversity = np.sum(rejections) \
    - (gamma)*0.5*np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,\
                                                  rejections.astype(bool)])

  metrics = [diversity, total_time, across_mc_total_time_solving, block_indexer, indexer]

  # with open(f"sharpe_results/rejections_c{couple}_s{setting}_j{job}.csv", "at") as file:
  #     file.write(",".join(map(str, rejections)) + "\n")

  with open(f"markowitz_cvxpy_results/metrics_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, metrics)) + "\n")

  # with open(f"sharpe_results/vanilla_rejections_c{couple}_s{setting}_j{job}.csv", "at") as file:
  #     file.write(",".join(map(str, vanilla_rejections)) + "\n")
