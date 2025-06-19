import numpy as np
import sys
import os
import time
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dacs_core.vanillaBH import bh
from dacs_core.optimizers import markowitz_eValueOptimizer_returnx

gammas = [0.015, 0.02, 0.025]

variants = []
for gamma_indexer in range(3):
    for job in range(250):
        variants.append((job, gamma_indexer))

start = time.time()

# use one command-line argument to get the variant
variant_block = int(sys.argv[1])
couple = True


unpermuted_similarity_matrix = np.load('similarityMatrix.npy')
for variant in tqdm(np.arange(variant_block*3, min((variant_block+1)*3,750))):
    job, gamma_indexer = variants[variant]
    gamma = gammas[gamma_indexer]
    np.random.seed(variant%250)
    
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
    skip=30
    num_mc_samples = 100
    num_mc_samples_per_step = 25
    diversities = []

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
                if (n+1.)/(1.+numCalibBelow) >= m/(alpha*max(1.,numTestBelow)):
                    x = markowitz_eValueOptimizer_returnx(eValues, similarityMatrix[n:,n:], \
                                                        alpha, gamma, m, 0.)

                    for b in range(num_mc_samples_per_step):
                        rejections = np.random.binomial(1,p=np.clip(x,0.,1.))
                        
                        diversity = np.sum(rejections) \
                      - gamma*0.5*np.sum((similarityMatrix[n:][:,n:])[rejections.astype(bool)][:,\
                                                  rejections.astype(bool)])

                        diversities.append(diversity)

        # sample a set randomly

        numCalibBelow -= int(argsortedS[indexer] < n)
        numTestBelow -= int(argsortedS[indexer] >= n)

    if reached_BH_stop == False:
        diversities = [0.]
    
    with open(f"optimized_markowitz_histogram_results/histogram_c{couple}_j{job}_q{quantile_indexer}_g{gamma_indexer}_a{int(100*alpha)}.csv", "at") as file:
        file.write(",".join(map(str, diversities)) + "\n")

end = time.time()
