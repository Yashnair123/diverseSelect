import numpy as np
import sys
import os
from dgps import dgp
from scores import mu_hat
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
import time
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
m_sizes = np.linspace(50,500,10).astype(int)
n_sizes = np.linspace(10,100,10).astype(int)
skip_sizes = np.linspace(5,50,10).astype(int)

for variant in range(3*variant_block, 3*(variant_block+1)):
    total_times = []
    across_mc_total_time_solvings = []
    ml_alg_ind = 1
    job, setting, alpha_ind = variants[variant]
    warm_or_custom, couple = True, True

    np.random.seed(variant)
    noise = 1.

      
    def get_scores(X, Y, mu_hat):
      return np.where(Y > 0, np.inf, -mu_hat.predict(X))

    for size_indexer in range(10):
        m = m_sizes[size_indexer]
        n = n_sizes[size_indexer]
        skip = skip_sizes[size_indexer]

        train = 1000
        # n = 500
        # m = 100
        alpha = [0.05, 0.2, 0.35][alpha_ind]
        # skip=50
        num_mc_samples = 300

        trainX, trainY = dgp(train, noise, setting)
        calibX, calibY = dgp(n, noise, setting)
        testX, testY = dgp(m, noise, setting)

        muHat = mu_hat(ml_alg_ind, trainX, trainY)
        calibS = get_scores(calibX, calibY, muHat)
        testS = get_scores(testX, np.zeros(m), muHat)
        
        train_labels = (trainY <= 0)

        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20
        )


        clf.fit(trainX, train_labels)


        combinedX = np.concatenate((calibX, testX))
        similarityMatrix = rbf_kernel(combinedX, combinedX)
        class_index = list(clf.classes_).index(True)
        probs = clf.predict_proba(combinedX)[:, class_index]
        log_likelihoods = np.log(probs)

        start = time.time()
        rejections, block_indexer, indexer, across_mc_total_time_solving = \
                    sharpe_approx_diverseSelect(calibS, testS, n, m, alpha, similarityMatrix, \
                                                num_mc_samples, couple, skip, warm_or_custom, log_likelihoods=log_likelihoods)
        end = time.time()

        total_time = end-start
        total_times.append(total_time)
        across_mc_total_time_solvings.append(across_mc_total_time_solving)


    #     vanilla_rejections, _, __ = bh(calibS, testS, n, m, alpha)

    #     diversity = np.sum(rejections)/np.sqrt(\
    #     np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,rejections.astype(bool)]))\
    #     if np.sum(rejections) > 0 else 0.

    #     fdp = np.sum([int(testY[i] <= 0)*int(rejections[i] == 1.) \
    #                         for i in range(m)])/max(1., np.sum(rejections))
    #     tdp = np.sum([int(testY[i] > 0)*int(rejections[i] == 1.) \
    #                     for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
    #                                                     for i in range(m)]))
    #     num_rejections = np.sum(rejections)

    #     pi0 = float(np.mean((trainY <= 0).astype(int)))

    #     metrics = [fdp, tdp, num_rejections, diversity, total_time, block_indexer, indexer]

    #     vanilla_diversity = np.sum(vanilla_rejections)/np.sqrt(\
    #     np.sum((similarityMatrix[n:][:,n:])[vanilla_rejections.astype(bool)][:,vanilla_rejections.astype(bool)]))\
    #     if np.sum(vanilla_rejections) > 0 else 0.

    #     vanilla_fdp = np.sum([int(testY[i] <= 0)*int(vanilla_rejections[i] == 1.) \
    #                         for i in range(m)])/max(1., np.sum(vanilla_rejections))
    #     vanilla_tdp = np.sum([int(testY[i] > 0)*int(vanilla_rejections[i] == 1.) \
    #                     for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
    #                                                     for i in range(m)]))
    #     vanilla_num_rejections = np.sum(vanilla_rejections)

    #     vanilla_metrics = [vanilla_fdp, vanilla_tdp, vanilla_num_rejections, vanilla_diversity]

    #     print(diversity, vanilla_diversity)
    #     print(total_time)

    with open(f"sharpe_scale_results/times_v{variant}.csv", "at") as file:
        file.write(",".join(map(str, total_times)) + "\n")
    with open(f"sharpe_scale_results/across_mc_solving_times_v{variant}.csv", "at") as file:
        file.write(",".join(map(str, across_mc_total_time_solvings)) + "\n")

    # with open(f"sharpe_results/metrics_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
    #     file.write(",".join(map(str, metrics)) + "\n")

    # with open(f"sharpe_results/solver_times_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
    #     file.write(",".join(map(str, [across_mc_total_time_solving,total_time])) + "\n")



    # with open(f"sharpe_results/vanilla_metrics_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
    #     file.write(",".join(map(str, vanilla_metrics)) + "\n")

    # pi_0arr = [pi0]

    # with open(f"sharpe_results/pi0s_v{variant}_w{warm_or_custom}_c{couple}.csv", "at") as file:
    #     file.write(",".join(map(str, pi_0arr)) + "\n")