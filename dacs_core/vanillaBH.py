import numpy as np

def bh(calibS, testS, n, m, alpha):
    combinedS = np.concatenate((calibS, testS))
    argsortedS = np.argsort(combinedS)

    numCalibRejections = n
    numTestRejections = m
    FDP_hat = ((1.+numCalibRejections)/max(1.,numTestRejections))*(m/(n+1.))

    indexer = n+m-1
    while FDP_hat > alpha and indexer >= 0:
        numCalibRejections -= int(argsortedS[indexer] < n)
        numTestRejections -= int(argsortedS[indexer] >= n)
        FDP_hat = ((1.+numCalibRejections)/max(1.,numTestRejections))*(m/(n+1.))
        indexer -= 1

    print(f'BH stopping index: {indexer}')
    if indexer < 0:
        vanillaRejections = np.zeros(m)
        return vanillaRejections, -1, -1
    
    vanillaRejections = (testS <= combinedS[argsortedS[indexer]]).astype(float)

    num_calibAbove = n-numCalibRejections
    return vanillaRejections, indexer, num_calibAbove