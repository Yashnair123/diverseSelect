import numpy as np
import sys
import os
from dgp import dgp

from sklearn.metrics.pairwise import rbf_kernel
from scores import mu_hat
import time
from scipy.stats import norm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import markowitz_approx_diverseSelect

gammas = [0.075, 0.1, 0.125]
alphas = [0.05, 0.2, 0.35]
variants = []
variant_counter = 0
for gamma_indexer in range(3):
  for job in range(250):
    for use_likelihoods in [0,1,2]:
      for alpha_ind in range(3):
        variants.append((job, use_likelihoods, gamma_indexer, alpha_ind))
        variant_counter += 1

# use one command-line argument to get the variant
variant_block = int(sys.argv[1])

for variant in np.arange(variant_block*5, (variant_block+1)*5):
  ml_alg_ind = 1
  job, use_likelihoods, gamma_indexer, alpha_ind = variants[variant]
  print(job, use_likelihoods, gamma_indexer, alpha_ind)


  couple = True 
  gamma = gammas[gamma_indexer]
  np.random.seed([job, use_likelihoods, alpha_ind])


  noise = 1.

  # not the difference from usual setup here:
  # we are just operating on the mu's directly (i.e., since it's the
  # same mu as in DGP, we essentially have oracle access to the true regression funcion,
  # again, this is just to be able to analytically calculate true likelihood)
  def get_scores(X, Y, mu_hat):
    return np.where(Y > 0, np.inf, -mu_hat.predict(X))

  train = 1000
  n = 500
  m = 100

  alpha = alphas[alpha_ind]
  skip=50
  num_mc_samples = 300


  trainX, trainY = dgp(train, noise)
  calibX, calibY = dgp(n, noise)
  testX, testY = dgp(m, noise)

  muHat = mu_hat(trainX, trainY)
  calibS = get_scores(calibX, calibY, muHat)
  testS = get_scores(testX, np.zeros(m), muHat)

  combinedX = np.concatenate((calibX, testX))
  similarityMatrix = rbf_kernel(combinedX, combinedX)


  if use_likelihoods == 1:
    # estimate likelihoods using logistic regression
    from sklearn.linear_model import LogisticRegression
    # train classifier to learn log likelihoods
    clf = LogisticRegression()
    clf.fit(trainX, trainY > 0)
    log_likelihoods = clf.predict_log_proba(combinedX)[:,0]
  elif use_likelihoods == 2:
    # use true likelihoods
    log_likelihoods = norm.logcdf(-combinedX.sum(axis=1)/(np.sqrt(5)*noise))
  else:
     log_likelihoods = None

  start = time.time()
  print(gamma)
  rejections, block_indexer, indexer, _ = markowitz_approx_diverseSelect(calibS, testS, n, m, alpha, gamma, \
                                      similarityMatrix, num_mc_samples, couple, skip,True, True, log_likelihoods=log_likelihoods)
  end = time.time()

  total_time = end-start


  vanilla_rejections, vanilla_indexer, __ = bh(calibS, testS, n, m, alpha)

  diversity = np.sum(rejections) \
    - (gamma)*0.5*np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,\
                                                  rejections.astype(bool)])

  fdp = np.sum([int(testY[i] <= 0)*int(rejections[i] == 1.) \
                        for i in range(m)])/max(1., np.sum(rejections))
  tdp = np.sum([int(testY[i] > 0)*int(rejections[i] == 1.) \
                  for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                  for i in range(m)]))
  num_rejections = np.sum(rejections)

  pi0 = float(np.mean((trainY <= 0).astype(int)))
  metrics = [fdp, tdp, num_rejections, diversity, total_time, block_indexer, indexer]

  vanilla_diversity = np.sum(vanilla_rejections) \
    - (gamma)*0.5*np.sum((similarityMatrix[n:][:,n:])[vanilla_rejections.astype(bool)][:,\
                                                  vanilla_rejections.astype(bool)])

  vanilla_fdp = np.sum([int(testY[i] <= 0)*int(vanilla_rejections[i] == 1.) \
                        for i in range(m)])/max(1., np.sum(vanilla_rejections))
  vanilla_tdp = np.sum([int(testY[i] > 0)*int(vanilla_rejections[i] == 1.) \
                  for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                  for i in range(m)]))
  vanilla_num_rejections = np.sum(vanilla_rejections)

  vanilla_metrics = [vanilla_fdp, vanilla_tdp, vanilla_num_rejections, vanilla_diversity]


  with open(f"markowitz_results/metrics_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, metrics)) + "\n")


  with open(f"markowitz_results/vanilla_metrics_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, vanilla_metrics)) + "\n")

  pi_0arr = [pi0]
  with open(f"markowitz_results/pi0s_v{variant}.csv", "at") as file:
      file.write(",".join(map(str, pi_0arr)) + "\n")

  print(total_time)
  print(rejections.sum(), vanilla_rejections.sum())
  print(f'ours: {diversity}, vanilla: {vanilla_diversity}')