import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from ostUtils import constructVs, cluster_getExactYs, sharpe_coupled_mcApproximateYs,\
sharpe_uncoupled_mcApproximateYs, markowitz_coupled_mcApproximateYs,\
markowitz_uncoupled_mcApproximateYs
import dacs

from vanillaBH import bh

from optimizers import sharpe_eValueOptimizer, markowitz_eValueOptimizer, sharpe_eValueOptimizer_returnx
from tqdm import tqdm
import cvxpy as cp
import time


default_tol=1e-6

def cluster_exact_diverseSelect(calibS, # calibration scores
                  testS, # test scores
                  n, # num calibration
                  m, # num test
                  alpha, # nominal level
                  componentPredictions, # clusters (in same order as combinedScores)
                  numComponents, # number of total components
                  skip=1 # how much to skip by, if using skipper
                  ):
    '''
    Run the method for clustering based objective,
    given component predictions
    '''
    # first get bh stopping time index
    bh_rejections, bh_st, bh_numCalibAbove = bh(calibS, testS, n, m, alpha)


    # if BH returns nothing, we can't return anything
    if np.all(bh_rejections == 0.):
        return np.zeros(m)
    
    bh_numCalibAbove = (n-np.cumsum(np.argsort(np.concatenate((calibS, testS))) < n))[bh_st]

    # get combined scores and sort them
    combinedS = np.concatenate((calibS, testS))
    argsortedS = np.argsort(combinedS)

    # get rewards (Ys) and Snell envelope (Vs)
    R = cluster_getExactYs(bh_st, bh_numCalibAbove, combinedS, componentPredictions,\
                            numComponents, n, m, alpha)
    # poss_num_vals_j = n+1-bh_numCalibAbove
    # initialize Ys

    V = constructVs(bh_st, bh_numCalibAbove, R, n, m, skip)

    indexer = bh_st
    # get indicator array of being calib for sorted scores
    B_array = (argsortedS < n).astype(int)

    # this gets reverse cumsum (i.e., S_array[i] counts
    # number of calib weakly above ith)
    num_calibBelow_array = np.cumsum(B_array)
    S_array = n-num_calibBelow_array
    
    # initialize relevant filtration variables at time n+m-1
    Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
    Si = S_array[indexer] #number of calib lying strictly above
    numCalibRejections = n-Si # numCalibBelow at first index value

    y = R[indexer,Si-bh_numCalibAbove]
    v = V[indexer,Si-bh_numCalibAbove]

    # stop once v=y
    dontStop = (v > y)
    while dontStop:
        indexer -= skip
        indexer = max(0,indexer) # will be relevant if skip does not evenly divide N
        Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
        Si = S_array[indexer] #number of calib lying weakly above
        
        # this is number of calib lying weakly below
        numCalibRejections = n-Si
        y = R[indexer,Si-bh_numCalibAbove]
        v = V[indexer,Si-bh_numCalibAbove]


        dontStop = (v > y)
    
    print(f'Stopping index: {indexer}')
    # Now that we have stopped at the stopping time, solve the e-value optimization
    # program

    # get indices which we can feasibly reject and their corresponding components
    posEvalueIndices = np.where(testS <= combinedS[argsortedS[indexer]])[0]+n
    posEComponents = componentPredictions[posEvalueIndices]
    
    # obtain the value of EValue and min num of allowable rejections
    eValueMultiplier = (n+1.)/(1.+numCalibRejections)
    rejectionLowerBound = np.ceil(m/(alpha*eValueMultiplier))

    # if infeasible, return no rejection
    if len(posEvalueIndices) < rejectionLowerBound:
        return np.zeros(m)


    # get counter array for number of of poss rejections
    # in each component
    num_per_component = np.bincount(posEComponents, minlength=numComponents)
    
    # go through list in sorted order and greedily take most of each item possible
    # if below the threshold, otherwise satisfy the threshold
    sortedNumComponents = np.sort(num_per_component)
    argsortedNumComponents = np.argsort(num_per_component)

    argSortedNumComponentSelection = np.zeros(numComponents)

    if sortedNumComponents[0] * numComponents >= rejectionLowerBound:
        # take at least sortedNumComponents[0] from each component, and exactly
        # rejectionLowerBound in total
        argSortedNumComponentSelection = sortedNumComponents[0]

    else:
        updatedLowerBound = rejectionLowerBound
        for indexer in range(numComponents):
            numRemainingComponents = numComponents-indexer
            if sortedNumComponents[indexer]*numRemainingComponents\
                >= updatedLowerBound:
                # tentative solution
                argSortedNumComponentSelection[indexer:] = sortedNumComponents[indexer]
                while np.sum(argSortedNumComponentSelection[indexer:]) > updatedLowerBound:
                    argmax_ind = np.argmax(argSortedNumComponentSelection[indexer:])
                    (argSortedNumComponentSelection[indexer:])[argmax_ind] -= 1
                break
            else:
                argSortedNumComponentSelection[indexer] = sortedNumComponents[indexer]
                updatedLowerBound -= sortedNumComponents[indexer]

    numComponentSelections = np.zeros(numComponents)
    
    numComponentSelections[argsortedNumComponents] = argSortedNumComponentSelection

    
    # now that we have decided how many from each component we are willing to take
    # iterate through the possible rejections to select which indices exactly to 
    # reject
    rejection_inds = []
    component_counter = np.zeros(numComponents)
    for i in range(len(posEvalueIndices)):
        if component_counter[componentPredictions[posEvalueIndices[i]]] \
            < numComponentSelections[componentPredictions[posEvalueIndices[i]]]:
            
            component_counter[componentPredictions[posEvalueIndices[i]]] += 1
            rejection_inds.append(posEvalueIndices[i])

    # shift back so that the rejection inds are in {0,...,m-1}
    rejection_inds = np.array(rejection_inds).astype(int)-n
    
    # use masking to construct indicator rejection array
    rejections = np.zeros(m)
    rejections[rejection_inds] = 1

    return rejections


def sharpe_approx_diverseSelect(calibS, # calibration scores
                  testS, # test scores
                  n, # num calib data
                  m, # num test data
                  alpha, # nominal level
                  similarityMatrix, # (kernel) similarity matrix
                  num_mc_samples, # number of MC samples to get for Ys
                # E[sharpeRatio(b_*)] for b_* ~ Bern(x^*) after we have
                # already solved for the optimal x^* (this MC is cheap)
                  couple=True, # whether or not to use coupled sampler
                  skip=1, # how many rows to skip,
                  warm_or_custom = True,
                  use_mosek_override=True
                  ):
    '''
    Similarity matrix must be computed for all n+m points
    '''
    # first get bh stopping time index
    bh_rejections, bh_st, bh_numCalibAbove = bh(calibS, testS, n, m, alpha)

    # if BH returns nothing, we can't return anything
    if np.all(bh_rejections == 0.):
        return np.zeros(m), -1, -1, 0.
    
    # redefine skip:
    skip = max(int(np.ceil((bh_st+1)/skip)),1)

    # get combined scores and sort them
    combinedS = np.concatenate((calibS, testS))
    argsortedS = np.argsort(combinedS)
    
    N = n+m
    poss_num_vals_j = n+1-bh_numCalibAbove
    # initialize Ys
    R = np.zeros((bh_st+1, poss_num_vals_j))
    # get rewards (Ys) and Snell envelope (Vs)
    
    print("Monte Carlo approximating rewards...")

    start = time.time()
    across_mc_total_time_solving = 0
    for _ in tqdm(range(num_mc_samples)):
        if couple:
            mcR, total_time_solving = sharpe_coupled_mcApproximateYs(bh_st, bh_numCalibAbove, combinedS, similarityMatrix, \
                                                 n, m, alpha, skip, warm_starting=warm_or_custom)
        else:
            mcR, total_time_solving = sharpe_uncoupled_mcApproximateYs(bh_st, bh_numCalibAbove, combinedS, similarityMatrix, \
                                                   n, m, alpha, skip, custom=warm_or_custom)
        across_mc_total_time_solving += total_time_solving

        R += mcR/num_mc_samples

    end = time.time()

    print(f"Total: {end-start} vs {across_mc_total_time_solving}")
    
    start = time.time()
    V = constructVs(bh_st, bh_numCalibAbove, R, n, m, skip)

    indexer = bh_st
    
    # get indicator array of being calib for sorted scores
    B_array = (argsortedS < n).astype(int)

    # this gets reverse cumsum (i.e., S_array[i] counts
    # number of calib weakly above ith)
    num_calibBelow_array = np.cumsum(B_array)
    S_array = n-num_calibBelow_array
    
    # initialize relevant filtration variables at time n+m-1
    Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
    Si = S_array[indexer] #number of calib lying weakly above
    numCalibRejections = n-Si # numCalibBelow at first index value

    y = R[indexer,Si-bh_numCalibAbove]
    v = V[indexer,Si-bh_numCalibAbove]


    number_of_blocks = int(np.ceil(bh_st/skip))
    block_indexer = number_of_blocks
    dontStop = (v > y)
    while dontStop:
        block_indexer -= 1
        indexer -= skip
        indexer = max(0,indexer) # will be relevant if skip does not evenly divide N
        Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
        Si = S_array[indexer] #number of calib lying strictly above
        
         # this is number of calib lying weakly below
        numCalibRejections = n-Si
        y = R[indexer,Si-bh_numCalibAbove]
        v = V[indexer,Si-bh_numCalibAbove]

        dontStop = (v > y)

    print(f'Stopping index: {indexer}')

    if indexer < 0:
        print("Error: Snell envelope incorrectly defined")
    else:
        eValues = (testS <= combinedS[argsortedS[indexer]]).astype(float) * \
            (n+1.)/(1.+numCalibRejections)
        
        if np.count_nonzero(eValues) == 0:
            rejections = np.zeros(m)
        elif np.max(eValues) < m/(alpha*np.count_nonzero(eValues)): 
            rejections = np.zeros(m)
        else:
            if (couple == True or (couple == False and warm_or_custom == True)) and not use_mosek_override:
                mask = (eValues > 0).astype(bool)
                solver = dacs.SharpePGDSolver(S=(similarityMatrix[n:,n:])[mask][:,mask], C=(alpha/m)*(n+1.)/(1.+numCalibRejections),
                                                    abs_tol=default_tol,
                                                    rel_tol=default_tol)
                xopt = solver.solve()

            else:
                mask = (eValues > 0).astype(bool)
                numActive = mask.sum()
                y = cp.Variable(numActive)
                prob = cp.Problem(cp.Minimize((cp.quad_form(y, cp.psd_wrap((similarityMatrix[n:,n:])[mask][:,mask])))),
                            [0 <= y,
                            y <= (alpha/m)*(n+1.)/(1.+numCalibRejections), #(self-consistency constraint simplifies 
                            # under assumption that 1^\top y = 1)
                            np.ones(numActive).T @ y == 1
                            ])
                prob.solve(verbose=False, solver=cp.MOSEK)
                xopt = y.value

            xopt = xopt * (1./np.max(xopt))
            rejections = np.zeros(m)
            rejections[mask] = np.random.binomial(1,p=np.clip(xopt,0.,1.))
    end = time.time()

    print(f'Remainder: {end-start}')
    return rejections, block_indexer, indexer, across_mc_total_time_solving


def markowitz_approx_diverseSelect(calibS, # calibration scores
                  testS, # test scores
                  n, # num calib data
                  m, # num test data
                  alpha, # nominal level
                  gamma, # lagrangian parameter
                  similarityMatrix, # (kernel) similarity matrix
                  num_mc_samples, # number of MC samples to get for Ys
                # E[sharpeRatio(b_*)] for b_* ~ Bern(x^*) after we have
                # already solved for the optimal x^* (this MC is cheap)
                  couple=True, # whether or not to use coupled sampler (faster)
                  skip=1, # how many rows to skip
                  warm_or_custom = True,
                  use_mosek_override=True
                  ):
    '''
    Similarity matrix must be computed for all n+m points
    '''
    # first get bh stopping time index
    bh_rejections, bh_st, bh_numCalibAbove = bh(calibS, testS, n, m, alpha)

    # redefine skip:
    skip = max(int(np.ceil((bh_st+1)/skip)),1)

    # if BH returns nothing, we can't return anything
    if np.all(bh_rejections == 0.):
        return np.zeros(m), -1, -1, 0.

    # get combined scores and sort them
    combinedS = np.concatenate((calibS, testS))
    argsortedS = np.argsort(combinedS)
    
    N = n+m
    poss_num_vals_j = n+1-bh_numCalibAbove
    # initialize Ys
    R = np.zeros((bh_st+1, poss_num_vals_j))
    # get rewards (Ys) and Snell envelope (Vs)
    
    across_mc_total_time_solving = 0
    print("Monte Carlo approximating rewards...")
    for _ in tqdm(range(num_mc_samples)):
        if couple:
            mcR, total_time_solving = markowitz_coupled_mcApproximateYs(bh_st, bh_numCalibAbove, gamma, \
                                combinedS, similarityMatrix, n, m, alpha, skip, warm_starting=warm_or_custom)
        else:
            mcR, total_time_solving = markowitz_uncoupled_mcApproximateYs(bh_st, bh_numCalibAbove, gamma, \
                                combinedS, similarityMatrix, n, m, alpha, skip, custom=warm_or_custom)
        R += mcR/num_mc_samples
        across_mc_total_time_solving += total_time_solving


    V = constructVs(bh_st, bh_numCalibAbove, R, n, m, skip)
    

    indexer = bh_st
    
    # get indicator array of being calib for sorted scores
    B_array = (argsortedS < n).astype(int)

    # this gets reverse cumsum (i.e., S_array[i] counts
    # number of calib strictly above ith)
    num_calibBelow_array = np.cumsum(B_array)
    S_array = n-num_calibBelow_array
    
    # initialize relevant filtration variables at time n+m-1
    Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
    Si = S_array[indexer] #number of calib lying weakly above
    numCalibRejections = (n-Si) # numCalibBelow at first index value

    y = R[indexer,Si-bh_numCalibAbove]
    v = V[indexer,Si-bh_numCalibAbove]


    number_of_blocks = int(np.ceil(bh_st/skip))
    block_indexer = number_of_blocks
    dontStop = (v > y)
    while dontStop:
        block_indexer -= 1
        indexer -= skip
        indexer = max(0,indexer) # will be relevant if skip does not evenly divide N
        Bi = int(argsortedS[indexer] < n) #bernoulli incidator of beign calib
        Si = S_array[indexer] #number of calib lying weakly above
        
         # this is number of calib lying weakly below
        numCalibRejections = n-Si
        y = R[indexer,Si-bh_numCalibAbove]
        v = V[indexer,Si-bh_numCalibAbove]

        dontStop = (v > y)

    print(f'Stopping index: {indexer}')

    if indexer < 0:
        print("Error: Snell envelope incorrectly defined")
    else:
        eValues = (testS <= combinedS[argsortedS[indexer]]).astype(float) * \
            (n+1.)/(1.+numCalibRejections)
        
        if np.count_nonzero(eValues) == 0:
            rejections = np.zeros(m)
        elif np.max(eValues) < m/(alpha*np.count_nonzero(eValues)): 
            rejections = np.zeros(m)
        else:
            if (couple == True or (couple == False and warm_or_custom == True)) and not use_mosek_override:
                mask = (eValues > 0).astype(bool)
                solver = dacs.MarkowitzPGDSolver(S=(similarityMatrix[n:,n:])[mask][:,mask],
                                                C=(alpha/m)*(n+1.)/(1.+numCalibRejections), gamma=gamma,
                                                    abs_tol=default_tol,
                                                    rel_tol=default_tol)
                            
                xopt = solver.solve()
            else:
                mask = (eValues > 0).astype(bool)
                numActive = mask.sum()
                y = cp.Variable(numActive)
                prob = cp.Problem(cp.Minimize((gamma*0.5*cp.quad_form(y, cp.psd_wrap((similarityMatrix[n:,n:])[mask][:,mask]))) \
                                                - np.ones(numActive).T @ y),
                                [0 <= y,
                                y <= (alpha/m)*(n+1.)/(1.+numCalibRejections) * (np.ones(numActive).T @ y),
                                y <= 1
                                ])
                    
                prob.solve(verbose=False, solver=cp.MOSEK)
                xopt = y.value

            rejections = np.zeros(m)
            rejections[mask] = np.random.binomial(1,p=np.clip(xopt,0.,1.))
        

    return rejections, block_indexer, indexer, across_mc_total_time_solving