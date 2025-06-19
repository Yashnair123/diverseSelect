import time
import numpy as np
import sys
import os
from dgps import dgp
from scores import mu_hat
from sklearn.metrics.pairwise import rbf_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import sharpe_approx_diverseSelect


variants = []
variant_counter = 0
for job in range(250):
  for setting in range(2):
    for alpha_ind in range(3):
      variants.append((job, setting, alpha_ind))
      variant_counter += 1

# use one command-line argument to get the variant
variant_block = int(sys.argv[1])

for variant in range(2*variant_block, 2*(variant_block+1)):
    ml_alg_ind = 1
    job, setting, alpha_ind = variants[variant]
    couple = False
    custom = False

    np.random.seed(variant)


    noise = 1.

    
    def get_scores(X, Y, mu_hat):
        return np.where(Y > 0, np.inf, -mu_hat.predict(X))

    train = 1000
    n = 500
    m = 100
    alpha = [0.05, 0.2, 0.35][alpha_ind]
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


    start = time.time()
    rejections, block_indexer, indexer, across_mc_total_time_solving = \
                            sharpe_approx_diverseSelect(calibS, testS, n, m, alpha, similarityMatrix, \
                                            num_mc_samples, couple, skip, custom)
    end = time.time()

    diversity = np.sum(rejections)/np.sqrt(\
    np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,rejections.astype(bool)]))\
    if np.sum(rejections) > 0 else 0.
    total_time = end-start


    metrics = [diversity, total_time, across_mc_total_time_solving, block_indexer, indexer]

    with open(f"sharpe_cvxpy_results/metrics_v{variant}.csv", "at") as file:
        file.write(",".join(map(str, metrics)) + "\n")