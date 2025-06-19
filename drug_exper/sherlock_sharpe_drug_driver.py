import numpy as np
import sys
import os
from sklearn.metrics.pairwise import rbf_kernel
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.diverseSelect import sharpe_approx_diverseSelect

print("HERE")

# use one command-line argument to get the variant
job = int(sys.argv[1])
couple = True

np.random.seed(job)

unpermuted_similarity_matrix = np.load('similarityMatrix.npy')
permutation = np.load(f'permutations/perm_j{750+job}.npy')
similarityMatrix = unpermuted_similarity_matrix[permutation][:,permutation]


testS = np.load(f'scores_and_Ys/testS_j{750+job}.npy')
quantile_indexer = 1
# MAKE SURE TO SAVE FINAL RESULS BY QUANTILE INDEXER
calibS = np.load(f'scores_and_Ys/calibS_j{750+job}_q{quantile_indexer}.npy')
testY = np.load(f'scores_and_Ys/testY_j{750+job}_q{quantile_indexer}.npy')

n = len(calibS)
m = len(testS)
alpha = 0.5
skip=100
num_mc_samples = 500

start = time.time()
rejections, block_indexer, indexer, _ = sharpe_approx_diverseSelect(calibS, testS, n, m, alpha, \
                                    similarityMatrix, num_mc_samples, couple, skip, True)
end = time.time()

total_time = end-start


vanilla_rejections, _, __ = bh(calibS, testS, n, m, alpha)

diversity = np.sum(rejections)/np.sqrt(\
np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,rejections.astype(bool)]))\
if np.sum(rejections) > 0 else 0.

fdp = np.sum([int(testY[i] <= 0)*int(rejections[i] == 1.) \
                    for i in range(m)])/max(1., np.sum(rejections))
tdp = np.sum([int(testY[i] > 0)*int(rejections[i] == 1.) \
                for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                for i in range(m)]))
num_rejections = np.sum(rejections)

metrics = [fdp, tdp, num_rejections, diversity, total_time, block_indexer, indexer]

vanilla_diversity = np.sum(vanilla_rejections)/np.sqrt(\
np.sum((similarityMatrix[n:][:,n:])[vanilla_rejections.astype(bool)][:,vanilla_rejections.astype(bool)]))\
if np.sum(vanilla_rejections) > 0 else 0.

vanilla_fdp = np.sum([int(testY[i] <= 0)*int(vanilla_rejections[i] == 1.) \
                    for i in range(m)])/max(1., np.sum(vanilla_rejections))
vanilla_tdp = np.sum([int(testY[i] > 0)*int(vanilla_rejections[i] == 1.) \
                for i in range(m)])/max(1., np.sum([int(testY[i] > 0) \
                                                for i in range(m)]))
vanilla_num_rejections = np.sum(vanilla_rejections)

vanilla_metrics = [vanilla_fdp, vanilla_tdp, vanilla_num_rejections, vanilla_diversity]



print(metrics)
print(vanilla_metrics)


with open(f"sharpe_results/metrics_c{couple}_j{job}_q{quantile_indexer}_a{int(100*alpha)}.csv", "at") as file:
    file.write(",".join(map(str, metrics)) + "\n")


with open(f"sharpe_results/vanilla_metrics_c{couple}_j{job}_q{quantile_indexer}_a{int(100*alpha)}.csv", "at") as file:
    file.write(",".join(map(str, vanilla_metrics)) + "\n")

print(total_time)