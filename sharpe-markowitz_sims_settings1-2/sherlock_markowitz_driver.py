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

variant_block = int(sys.argv[1])

for variant in np.arange(variant_block*5, (variant_block+1)*5):
  for (warm_or_custom, couple) in [(True, True), (False, True), (True, False)]:
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

    start = time.time()
    print(gamma)
    rejections, block_indexer, indexer, across_mc_total_time_solving \
                = markowitz_approx_diverseSelect(calibS, testS, n, m, alpha, gamma, \
                                        similarityMatrix, num_mc_samples, couple, skip, warm_or_custom)
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

    # with open(f"sharpe_results/rejections_c{couple}_s{setting}_j{job}.csv", "at") as file:
    #     file.write(",".join(map(str, rejections)) + "\n")

    with open(f"markowitz_results/metrics_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
        file.write(",".join(map(str, metrics)) + "\n")

    with open(f"markowitz_results/solver_times_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
        file.write(",".join(map(str, [across_mc_total_time_solving,total_time])) + "\n")

    with open(f"markowitz_results/vanilla_metrics_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
        file.write(",".join(map(str, vanilla_metrics)) + "\n")

    pi_0arr = [pi0]
    with open(f"markowitz_results/pi0s_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
        file.write(",".join(map(str, pi_0arr)) + "\n")
