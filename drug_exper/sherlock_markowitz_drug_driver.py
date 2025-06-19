import numpy as np
import sys
import os
from sklearn.metrics.pairwise import rbf_kernel
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import markowitz_approx_diverseSelect

print("HERE")
gammas = [0.015, 0.02, 0.025]
# use one command-line argument to get the variant
variant = int(sys.argv[1])
couple = True


variants = []
for gamma_indexer in range(3):
    for job in range(250):
        variants.append((job, gamma_indexer))

np.random.seed(variant%250)
job, gamma_indexer = variants[variant]
gamma = gammas[gamma_indexer]

unpermuted_similarity_matrix = np.load('similarityMatrix.npy')
permutation = np.load(f'permutations/perm_j{variant%250}.npy')
similarityMatrix = unpermuted_similarity_matrix[permutation][:,permutation]

testS = np.load(f'scores_and_Ys/testS_j{variant%250}.npy')
quantile_indexer = 1
# MAKE SURE TO SAVE FINAL RESULS BY QUANTILE INDEXER
calibS = np.load(f'scores_and_Ys/calibS_j{variant%250}_q{quantile_indexer}.npy')
testY = np.load(f'scores_and_Ys/testY_j{variant%250}_q{quantile_indexer}.npy')


n = len(calibS)
m = len(testS)
alpha = 0.5
skip=100
num_mc_samples = 500

start = time.time()
rejections, block_indexer, indexer, _ = markowitz_approx_diverseSelect(calibS, testS, n, m, alpha, gamma, \
                                    similarityMatrix, num_mc_samples, couple, skip)
end = time.time()

total_time = end-start


vanilla_rejections, _, __ = bh(calibS, testS, n, m, alpha)

diversity = np.sum(rejections) \
- gamma*0.5*np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,\
                                                rejections.astype(bool)])

fdp = np.sum([int(testY[i] <= 0)*int(rejections[i] == 1.) \
                    for i in range(m)])/max(1., np.sum(rejections))
tdp = np.sum([int(testY[i] > 0)*int(rejections[i] == 1.) \
                for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                for i in range(m)]))
num_rejections = np.sum(rejections)

metrics = [fdp, tdp, num_rejections, diversity, total_time, block_indexer, indexer]

vanilla_diversity = np.sum(vanilla_rejections) \
- gamma*0.5*np.sum((similarityMatrix[n:][:,n:])[vanilla_rejections.astype(bool)][:,\
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

with open(f"markowitz_results/metrics_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{int(100*alpha)}.csv", "at") as file:
    file.write(",".join(map(str, metrics)) + "\n")

# with open(f"sharpe_results/vanilla_rejections_c{couple}_s{setting}_j{job}.csv", "at") as file:
#     file.write(",".join(map(str, vanilla_rejections)) + "\n")

with open(f"markowitz_results/vanilla_metrics_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{int(100*alpha)}.csv", "at") as file:
    file.write(",".join(map(str, vanilla_metrics)) + "\n")

print(total_time)