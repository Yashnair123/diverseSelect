import numpy as np
import copy
import cvxpy as cp
import pandas as pd
import math
from scipy.stats import hypergeom, binom
import scipy as sp
import heapq
import time
import osqp
import scipy.sparse as sp
from scipy.linalg import sqrtm, cholesky
from tqdm import tqdm
import dacs

from optimizers import sharpe_eValueOptimizer, markowitz_eValueOptimizer

default_tol=1e-6


def check_impossible(i,j,n,bh_numCalibAbove,bh_st):
    '''
    A simple method to check if a certain value of the triple
    (i,j,k) is even possible
    '''
    return (j < max(bh_numCalibAbove, n-(i+1))) \
    or (j > min(n, bh_st+1-(i+1)+bh_numCalibAbove))

vectorized_check_impossible = np.vectorize(check_impossible)


def custom_multiply(a, b):
    '''
    Define a multiplication function under which
    infinity * 0 = 0
    '''
    # Check if either a or b is infinity and the other is zero
    if (np.isinf(a) and b == 0) or (np.isinf(b) and a == 0):
        return 0.
    else:
        return a * b

vectorized_custom_multiply = np.vectorize(custom_multiply)

# modified from stackoverflow
def convolve_many(arrays):
    """
    Convolution using FFT.
    """
    # Transform, take the product, and do the inverse transform
    # to get the convolution.
    ffts = np.fft.fft(arrays, axis=0)
    fft_of_convolution = ffts.prod(axis=1)
    convolution = np.fft.ifft(fft_of_convolution, axis=0)

    # Assuming real inputs, the imaginary part of the output can
    # be ignored.
    
    # clip to avoid numerical degenericies
    return np.clip(convolution.real,0.,1.)


def cluster_getExactYs(
    bh_st, # index at which bh stops
    bh_numCalibAbove, # number of calibration points weakly above at bh_st
    scores, # list of conformity scores
    componentPredictions, # cluster predictions for each datapoint (these 
    # should be 0-indexed)
    numComponents, # number of clusters
    n, # number of calibration
    m, # number of test
    alpha, # nominal level
    skip=1 # value of skipper to fill out every skip^th row of DP table
):
    '''
    Return matrix R for which R[i][j] is 
    the (expected) value of O_{(i+1)} when N_{(i+1)}^Above = j
    (all these plus ones are due to zero-indexing) 

    Reward is:
    1. -1/C points if no rejections can be made
    2. objective function points if 
       rejections can be made
    '''
    
    worst_case_value = -1./numComponents # value of empty rejection set
    poss_num_vals_j = n-bh_numCalibAbove+1


    print("Calculating measurable rewards...")
    # work with scores and their component predictions in sorted order
    sortedInds = np.argsort(scores)
    sorted_componentPredictions = componentPredictions[sortedInds]
    
    # construct cumsum 2d array for components
    sorted_counts = []
    for component_indexer in range(numComponents):
        sorted_count = np.cumsum(sorted_componentPredictions \
                                 == component_indexer)
        sorted_counts.append(sorted_count)
    
    # this stacks the component counts as columns (shape is (n+m)xC)
    sorted_componentCumsum = np.column_stack(sorted_counts)

    R = np.full((bh_st+1, poss_num_vals_j), np.nan)

    steps = list(range(bh_st, -1, -skip))
    
    # Ensure that 0 is included, even if the last step is not 
    # exactly at 0
    if steps[-1] != 0:
        steps.append(0)

    for i in tqdm(steps):
        # in case the minimum component value is 0, we can only get
        # 0 points for feasibility
        if np.min(sorted_componentCumsum[i]) == 0:
            for numActive in range(m+1):
                j = numActive-i-1+n

                # Rule out impossible cases (again)
                if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                    continue

                # check E-value infeasibility
                numCalibBelow = n-j
                eValueMultiplier = (n+1.)/(1.+numCalibBelow)
                
                # compute lower bound that self-consistency imposes
                rejectionLowerBound = np.ceil(m/(alpha*eValueMultiplier))

                # feasibility check
                if numActive >= rejectionLowerBound:
                    R[i,j-bh_numCalibAbove] = 0.
                else:
                    R[i,j-bh_numCalibAbove] = worst_case_value
        else:
            # this is the minimum possible value of j given stopped filtration
            min_j = max(bh_numCalibAbove, n-(i+1))
            
            # this is the max for rejectionLowerBound (K)
            max_rejectionLowerBound = np.ceil(m*(1.+n-min_j)/(alpha*(n+1.))).astype(int)


            # let s denote length of thresholds (i.e., this counts survival function
            # for the minimum multHyperGeom. Will need it to be computed for 1
            # up until min(max_rejectionLowerBound/numComponents, 
            # np.min(sorted_componentCumsum[i])))
            thresholds = np.arange(1,min(np.ceil(max_rejectionLowerBound/numComponents),\
                        np.min(sorted_componentCumsum[i]).astype(int))+1)
            
            # shape now is s x numComponents
            thresholds = np.tile(thresholds[:,np.newaxis], numComponents)
            
            # the -1 below is because survival function is defined 
            # strangely in scipy as (1-cdf)

            # Uses representation of multivariate hypergeometric as conditional
            # distribution of binomials given their sum

            # work with logprobs for numerical stability
            # the shape of below is 
            # s x numComponents because 
            # the shape of thresholds is s x numComponents
            # and shape of sorted_componentCumsum[i] is 
            # numComponents
            log_sfs = binom.logsf((thresholds-1), \
                                    sorted_componentCumsum[i], 0.5)


            # the shape of below is of size s (getting every bayes numerator term
            # for each possible value of threshold in the range)
            bayes_numerator_easy_terms = np.exp(log_sfs.sum(axis=1))
            
            # compute FFT
            # the sum of n_c's is t, and hence we zero-pad to be of length
            # t+1. So we compute the pmf values for each from 0, ..., t

            # size is (i+2) x C (reason it's i+2 is bc it's len(0, ..., i+1) since i+1=t)
            # this arrray will be used to construct pmf of truncated binomials
            marginal_clipped_log_pmfs = binom.logpmf(np.arange((i+1)+1)[:,np.newaxis], sorted_componentCumsum[i], 0.5)

            # mask is of shape (i+2) x C x s
            row_indices = np.arange(marginal_clipped_log_pmfs.shape[0])[:,np.newaxis,np.newaxis]
            mask = row_indices < thresholds.T # indices which should be 0 (violate conditioning
            # event)

            # marginal_clipped_log_pmfs is of shape 
            # (i+2) x C x s
            marginal_clipped_log_pmfs = np.where(mask, -np.inf, \
                                                    marginal_clipped_log_pmfs[:,:,np.newaxis])
            
            # divide by denom to get truncated pmf
            # shape of truncated pmfs is also (i+2) x C x s
            truncated_pmfs = np.exp(marginal_clipped_log_pmfs-log_sfs.T)

            # shape of convolution is (i+2) x s
            convolution = convolve_many(truncated_pmfs)

            # can now compute numerator of Bayes rule, shape (i+2) x s
            bayes_numerators = convolution * bayes_numerator_easy_terms
            
            # total sum of Binomial parameters across components
            # this will be needed to compute the Bayes denom
            # for various values of numActive
            total_n_sum = sorted_componentCumsum[i].sum()

            for numActive in range(m+1):
                j = numActive-i-1+n

                # Rule out impossible cases (again)
                if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                    continue

                numCalibBelow = n-j
                eValueMultiplier = (n+1.)/(1.+numCalibBelow)
                
                # compute lower bound that self-consistency imposes
                rejectionLowerBound = np.ceil(m/(alpha*eValueMultiplier))

                # feasibility check
                if numActive >= rejectionLowerBound:
                    bayes_denominator = binom.pmf(numActive, \
                                            total_n_sum, 0.5)
                    # probabilities is of length s
                    # 0th entry is 
                    # P(min >= 1) and last entry is P(min >= largest poss seeable
                    # value of rejectionLowerBound)
                    probabilities = bayes_numerators[numActive,:]/bayes_denominator

                    # want EV of the clipped min component. Specficically want 
                    # EV of min(min, ceil(K/C)), so we only use tail sum 
                    # formula for expectation up until ceil(K/C)
                    K_ts_t_overC = int(np.ceil(rejectionLowerBound/numComponents))
                    min_Nts = int(np.min(sorted_componentCumsum[i]))

                    # compute expectations:
                    if min_Nts < K_ts_t_overC:
                        indicator_expectation = 0./numComponents
                        # sum of probabilities[0], ..., probabilities[min_Nts-1]
                        # which is sum of survival(1) + ... + survival(min_Nts)
                        sum_expectation = probabilities[:min_Nts].sum()/rejectionLowerBound
                    else:
                        # probabilities[K_ts_t_overC-1] = survival(K_ts_t_overC)
                        indicator_expectation = probabilities[K_ts_t_overC-1]/numComponents

                        # probabilities[0] + ... + probabilities[K_ts_t_overC-2]
                        # = survival(1) + .... + survival(K_ts_t_overC-1)
                        sum_expectation = probabilities[:K_ts_t_overC-1].sum()/rejectionLowerBound

                    R[i,j-bh_numCalibAbove] = indicator_expectation + sum_expectation
                else:
                    R[i,j-bh_numCalibAbove] = worst_case_value

    return R


def generate_random_vector(numOnes, length, n, m):
    arr = np.empty(n+m, dtype=bool)
    arr[:numOnes] = True
    arr[numOnes:] = False
    np.random.shuffle(arr[:length])
    return arr

def sharpe_coupled_mcApproximateYs(
    bh_st, # index of BH stopping time
    bh_numCalibAbove, # number of calibration points weakly above at bh_st
    scores, # list of conformity scores; shape is (n+m,)
    similarityMatrix,  # (kenel) similarity matrix; shape is (n+m, n+m)
    n, # number of calibration
    m, # number of test
    alpha, # nominal level
    skip=1, # value of skipper to fill out every skip^th row of DP table
    bernMC = 50,
    warm_starting = True
):
    '''
    Return matrices Y0 and Y1 for which Yk[i][j] is the 
    (expected) value of Y_{(i+1)} when B_{(i+1)} = k
    (all these plus ones are due to zero-indexing) and S_{i+1} = j 
    (i.e., so it is really a  
    submatrix where j ranges from max(n-i+1,0) to min(n, N-(i+1)+1), inclusive)

    Important NB: this method generates a single (coupled) MC sample for each
    entry of the tables Y0 and Y1
    '''
    worst_case_value = 0.
    poss_num_vals_j = n+1-bh_numCalibAbove

    # get the indices and similarity matrix according to the sorted ordering
    sortedInds = np.argsort(scores)
    sortedSimilarityMatrix = np.ascontiguousarray(similarityMatrix[sortedInds][:,sortedInds])
    N = n+m
    
    # initialize approx reward matrices
    # Y0, Y1 = np.zeros((N, n+1)), np.zeros((N, n+1))
    R = np.full((bh_st+1, poss_num_vals_j), np.nan)

    steps = list(range(bh_st, -1, -skip))
    
    # Ensure that N-1 is included, even if the last step is not 
    # exactly at N-1
    if steps[-1] != 0:
        steps.append(0)

    total_time_solving = 0.
    numCalls = 0
    for numActive in range(m+1):
        warm = None
        # we will share the Monte Carlo sample for different values of i
        # but for the same numActive

        #### CHANGE ####
        # the set of activeTestInds will be coupled over i
        activeTestInds = np.array([])

        for i in steps:
            j = numActive-i-1+n
            # Rule out impossible cases
            if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                continue
            
            # this is the value to index the Y arrays
            j_reindexed = j-bh_numCalibAbove
            # in this case, it will be impossible to make rejections
            if numActive == 0.:
                R[i,j_reindexed] = worst_case_value#0.
                continue

            # operating under S_{i+1} = j,  
            # generate a single Monte Carlo sample to approx R_{i+1}.
            # First we construct the e-values
            # the below variable counts number of calibration weakly below the (i+1)th 
            numCalibBelow = n-j
            eValueMultiplier = (n+1.)/(1. + numCalibBelow)

            
            # At this stage, we randomly sample which of the points are test and
            # which are calibration.
            # Iff the number of calibration weakly below (i+1)th is numCalibBelow
            # then the number of test weakly below is i+1-numCalibBelow
            
            if i == 0 and numActive == 1:
                activeTestInds = np.array([0])
            else:
                # if we have not yet populated activeTestInds, we sample completely at
                # random
                if len(activeTestInds) == 0:
                    # select the numActive random test indices from {0,...,i}
                    activeTestInds = np.random.choice(i+1, numActive, replace=False)
                else:
                    # find indices which we cannot include
                    violationIndices = np.where(activeTestInds > i)[0]
                    
                    if len(violationIndices) > 0:
                        # these are the indices in {0,...,i-1} that are not yet
                        # in activeTestInds, but which we may randomly sample from
                        mask = np.ones(last_i+1, dtype=bool)
                        mask[activeTestInds] = False
                        mask[i+1:] = False
                        complement = np.where(mask)[0]
                        #complement = np.setdiff1d(np.arange(i+1), activeTestInds, assume_unique=True)
  
                        activeTestInds[violationIndices] \
                            = np.random.choice(complement, len(violationIndices), \
                                                replace=False)

            mcSimilarityMatrix = dacs.subset(sortedSimilarityMatrix,activeTestInds)
            last_i = i
            
            # in this edge case, the optimization is easily solved (without any pgd)
            if numActive == 1:
                # check self consistency:
                if eValueMultiplier >= m/alpha:
                    # in this case, just will accept:
                    # (technically, this assumes that all diagonal entries are same)
                    R[i,j_reindexed] = (1./mcSimilarityMatrix[0][0])
                else:
                    # if self-consistency violated incur 0 reward
                    R[i,j_reindexed] = worst_case_value
            
            # in non-trivial case, we must now use the pgd
            else:
                # stil check self-consistency
                if numActive * eValueMultiplier >= m/alpha:
                    if warm_starting == True:
                        start = time.time()
                        solver = dacs.SharpePGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier,
                                                abs_tol=default_tol,
                                                rel_tol=default_tol)
                        pgd_soln = solver.solve(warm)
                        end = time.time()
                        numCalls += 1
                    else:
                        start = time.time()
                        solver = dacs.SharpePGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier,
                                                abs_tol=default_tol,
                                                rel_tol=default_tol)
                        pgd_soln = solver.solve()
                        end = time.time()
                        numCalls += 1

                    total_time_solving += (end-start)
                    warm = pgd_soln
                    xopt = pgd_soln

                    # rescale so that largest x is 1
                    # (min(xopt) should be > 0 because we have already checked feasibility)
                    xopt = xopt * (1./np.max(xopt))
                    #nonzero = np.logical_or(np.isclose(xopt,np.ones_like(xopt)), np.isclose(xopt,np.zeros_like(xopt)))
                    #print(nonzero.sum()/nonzero.shape[0])
                    R[i,j_reindexed] = 0

                    p_clipped = np.clip(xopt, 0., 1.)
                    B = np.random.rand(bernMC, xopt.size) < p_clipped
            
                    R[i,j_reindexed] = dacs.mc_sharpe(mcSimilarityMatrix, B)
                    
                else:
                    # if self-consistency violated incur 0 reward
                    R[i,j_reindexed] = worst_case_value

    return R, total_time_solving

def sharpe_uncoupled_mcApproximateYs(
    bh_st, # index of BH stopping time
    bh_numCalibAbove, # number of calibration points weakly above at bh_st
    scores, # list of conformity scores; shape is (n+m,)
    similarityMatrix,  # (kenel) similarity matrix; shape is (n+m, n+m)
    n, # number of calibration
    m, # number of test
    alpha, # nominal level
    skip=1, # value of skipper to fill out every skip^th row of DP table
    bernMC = 50,
    custom = True
):
    '''
    Return matrices Y0 and Y1 for which Yk[i][j] is the 
    (expected) value of Y_{(i+1)} when B_{(i+1)} = k
    (all these plus ones are due to zero-indexing) and S_{i+1} = j 
    (i.e., so it is really a  
    submatrix where j ranges from max(n-i+1,0) to min(n, N-(i+1)+1), inclusive)

    Important NB: this method generates a single (uncoupled) MC sample for each
    entry of the tables Y0 and Y1
    '''

    worst_case_value = 0.
    poss_num_vals_j = n+1-bh_numCalibAbove

    # get the indices and similarity matrix according to the sorted ordering
    sortedInds = np.argsort(scores)
    sortedSimilarityMatrix = np.ascontiguousarray(similarityMatrix[sortedInds][:,sortedInds])

    N = n+m


    # initialize reward matrices
    # Y0, Y1 = np.zeros((N, n+1)), np.zeros((N, n+1))
    R = np.full((bh_st+1, poss_num_vals_j), np.nan)

    steps = list(range(bh_st, -1, -skip))
    
    # Ensure that N-1 is included, even if the last step is not 
    # exactly at N-1
    if steps[-1] != 0:
        steps.append(0)

    total_time_solving = 0.
    numCalls = 0
    for i in list(reversed(steps)):
        # first construct what the e-values would be if we were to stop at time i, depending
        # on the value of S_i
        for j in range(n+1):
            # operating under S_{i+1} = j, and that B_{(i+1)} = k, Monte Carlo approx R_{i+1}
            # first we construct the e-values
            # the below variable counts number of calibration weakly below the (i+1)th 
            numCalibBelow = n-j
            # the number of test points weakly below
            numActive = i + 1 - numCalibBelow

            # Rule out impossible cases
            if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                continue

            j_reindexed = j-bh_numCalibAbove
            
            eValueMultiplier = (n+1.)/(1. + numCalibBelow)
            
            # only need consider case when numActive is nonzero since otherwise, 
            # the value of the program is just 0
            if numActive != 0:
                # at this stage, we randomly sample which of the points are test and
                # which are calibration
                # if the number of calibration weakly below (i+1)th is numCalibBelow
                # then the number of test weakly below is i+1-numCalibBelow
                # if B_{(i+1)} = 1, then we cannot select i
                activeTestInds = np.random.choice(i+1, numActive, replace=False)
                
                # construct the randomly subsetted mcSimilarityMatrix
                mcSimilarityMatrix = dacs.subset(sortedSimilarityMatrix,activeTestInds)
                # construct eValue array
                if numActive > 0 and eValueMultiplier >= m/(alpha*numActive):
                    if custom:
                        start = time.time()
                        solver = dacs.SharpePGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier,
                                            abs_tol=default_tol,
                                            rel_tol=default_tol)
                        xopt = solver.solve()
                        end = time.time()
                        numCalls += 1

                    else:
                        start = time.time()
                        y = cp.Variable(numActive)
                        prob = cp.Problem(cp.Minimize((cp.quad_form(y, cp.psd_wrap(mcSimilarityMatrix)))),
                                    [0 <= y,
                                    y <= (alpha/m)*eValueMultiplier, #(self-consistency constraint simplifies 
                                    # under assumption that 1^\top y = 1)
                                    np.ones(numActive).T @ y == 1
                                    ])
                        prob.solve(verbose=False, solver=cp.MOSEK)
                        xopt = y.value
                        end = time.time()
                    
                    total_time_solving += (end-start)
                    xopt = xopt * (1./np.max(xopt))

                    p_clipped = np.clip(xopt, 0., 1.)
                    B = np.random.rand(bernMC, xopt.size) < p_clipped
            
                    programValue = dacs.mc_sharpe(mcSimilarityMatrix, B)
                else:
                    programValue = worst_case_value
                R[i,j_reindexed] = programValue
            else:
                R[i,j_reindexed] = worst_case_value # get reward 0 for self-consistency infeasible solution
    # print(f'Coupling: {total_time_coupling}, Solving: {total_time_solving}')
    return R, total_time_solving



def markowitz_coupled_mcApproximateYs(
    bh_st, # index of stopping time of BH
    bh_numCalibAbove, # number of calibration points weakly above at bh_st
    gamma, # lagrangian (convex combination) parameter
    scores, # list of conformity scores; shape is (n+m,)
    similarityMatrix,  # (kenel) similarity matrix; shape is (n+m, n+m)
    n, # number of calibration
    m, # number of test
    alpha, # nominal level
    skip=1, # value of skipper to fill out every skip^th row of DP table
    warm_starting = True
):
    '''
    Return matrices Y0 and Y1 for which Yk[i][j] is the 
    (expected) value of Y_{(i+1)} when B_{(i+1)} = k
    (all these plus ones are due to zero-indexing) and S_{i+1} = j 
    (i.e., so it is really a  
    submatrix where j ranges from max(n-i+1,0) to min(n, N-(i+1)+1), inclusive)

    Important NB: this method generates a single (coupled) MC sample for each
    entry of the tables Y0 and Y1
    '''
    rels = []
    worst_case_value = 0.
    poss_num_vals_j = n+1-bh_numCalibAbove

    # get the indices and similarity matrix according to the sorted ordering
    sortedInds = np.argsort(scores)
    sortedSimilarityMatrix = np.ascontiguousarray(similarityMatrix[sortedInds][:,sortedInds])
    N = n+m

    
    # initialize approx reward matrices
    # Y0, Y1 = np.zeros((N, n+1)), np.zeros((N, n+1))
    R = np.full((bh_st+1, poss_num_vals_j), np.nan)

    steps = list(range(bh_st, -1, -skip))
    
    # Ensure that N-1 is included, even if the last step is not 
    # exactly at N-1
    if steps[-1] != 0:
        steps.append(0)

    total_time_solving = 0
    numCalls = 0
    for numActive in range(m+1):
        warm = None
        # we will share the Monte Carlo sample for different values of i
        # but for the same numActive

        # the set of activeTestInds will be coupled over i
        activeTestInds = np.array([])
        for i in steps:
            j = numActive-i-1+n
            # Rule out impossible cases
            if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                continue

            j_reindexed = j-bh_numCalibAbove
            # in this case, it will be impossible to make rejections
            if numActive == 0.:
                R[i,j_reindexed] = worst_case_value
                continue

            # operating under S_{i+1} = j, and that B_{(i+1)} = k, 
            # generate a single Monte Carlo sample to approx R_{i+1}.
            # First we construct the e-values
            # the below variable counts number of calibration weakly below the (i+1)th 
            numCalibBelow = n-j
            eValueMultiplier = (n+1.)/(1. + numCalibBelow)

            
            # At this stage, we randomly sample which of the points are test and
            # which are calibration.
            # Iff the number of calibration weakly below (i+1)th is numCalibBelow
            # then the number of test weakly below is i+1-numCalibBelow
            
            if i == 0 and numActive == 1:
                activeTestInds = np.array([0])
            else:
                # if we have not yet populated activeTestInds, we sample completely at
                # random
                if len(activeTestInds) == 0:
                    # select the numActive random test indices from {0,...,i-1}
                    activeTestInds = np.random.choice(i+1, numActive, replace=False)
                else:
                    # find indices which we cannot include
                    violationIndices = np.where(activeTestInds > i)[0]
                    # otherwise, we do the coupling
                    if len(violationIndices) > 0:
                        # these are the indices in {0,...,i-1} that are not yet
                        # in activeTestInds, but which we may randomly sample from
                        mask = np.ones(last_i+1, dtype=bool)
                        mask[activeTestInds] = False
                        mask[i+1:] = False
                        complement = np.where(mask)[0]
                        #complement = np.setdiff1d(np.arange(i+1), activeTestInds, assume_unique=True)


                        activeTestInds[violationIndices] \
                            = np.random.choice(complement, len(violationIndices), \
                                                replace=False)
                        
            mcSimilarityMatrix = dacs.subset(sortedSimilarityMatrix,activeTestInds)
            last_i = i

            # in this edge case, the optimization is easily solved (without any pgd)
            if numActive == 1:
                # check self consistency:
                if eValueMultiplier >= m/alpha:
                    # in this case, just will accept:
                    # (technically, this assumes that all diagonal entries are same)
                    R[i,j_reindexed] = -(0.5*gamma-1)#-(0.5*gamma-1)
                else:
                    # if self-consistency violated incur 0 reward
                    R[i,j_reindexed] = worst_case_value
            
            # in non-trivial case, we must now use the pgd
            else:
                # stil check self-consistency
                # if self-consistency is exact, reject everything
                if numActive * eValueMultiplier == m/alpha:
                    R[i,j_reindexed]= -(0.5*(gamma)*mcSimilarityMatrix.sum()\
                                    -numActive)
                
                # in this case, there is strict feasibility
                elif numActive * eValueMultiplier > m/alpha:
                    if warm_starting:
                        start = time.time()
                        solver = dacs.MarkowitzPGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier, \
                                                        gamma=gamma,abs_tol=default_tol,
                                                        rel_tol=default_tol)
                        pgd_soln = solver.solve(warm)
                        end = time.time()

                    else:
                        start = time.time()
                        solver = dacs.MarkowitzPGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier, \
                                                        gamma=gamma,abs_tol=default_tol,
                                                        rel_tol=default_tol)
                        pgd_soln = solver.solve()
                        end = time.time()
                        numCalls += 1


                    total_time_solving += end-start

                    warm = pgd_soln
                    xopt = pgd_soln

                    xopt_squared = xopt**2
                    avgVal = xopt.sum() - 0.5*gamma*(xopt.T @ mcSimilarityMatrix @ xopt\
                    + np.dot(mcSimilarityMatrix.diagonal(), xopt-xopt_squared))
                    
                    R[i,j_reindexed] = avgVal
                else:
                    # if self-consistency violated incur 0 reward
                    R[i,j_reindexed] = worst_case_value
                    
    return R, total_time_solving


def markowitz_uncoupled_mcApproximateYs(
    bh_st, # index of BH stopping time
    bh_numCalibAbove, # number of calibration points weakly above at bh_st
    gamma, # lagrangian (convex combination) parameter
    scores, # list of conformity scores; shape is (n+m,)
    similarityMatrix,  # (kenel) similarity matrix; shape is (n+m, n+m)
    n, # number of calibration
    m, # number of test
    alpha, # nominal level
    skip=1, # value of skipper to fill out every skip^th row of DP table
    custom=True
):
    '''
    Return matrices Y0 and Y1 for which Yk[i][j] is the 
    (expected) value of Y_{(i+1)} when B_{(i+1)} = k
    (all these plus ones are due to zero-indexing) and S_{i+1} = j 
    (i.e., so it is really a  
    submatrix where j ranges from max(n-i+1,0) to min(n, N-(i+1)+1), inclusive)

    Important NB: this method generates a single (uncoupled) MC sample for each
    entry of the tables Y0 and Y1
    '''
    worst_case_value = 0.
    poss_num_vals_j = n+1-bh_numCalibAbove

    # get the indices and similarity matrix according to the sorted ordering
    sortedInds = np.argsort(scores)
    sortedSimilarityMatrix = np.ascontiguousarray(similarityMatrix[sortedInds][:,sortedInds])
    
    N = n+m

    # initialize reward matrices
    # Y0, Y1 = np.zeros((N, n+1)), np.zeros((N, n+1))
    R = np.full((bh_st+1, poss_num_vals_j), np.nan)

    steps = list(range(bh_st, -1, -skip))

    numCells = 0
    
    # Ensure that N-1 is included, even if the last step is not 
    # exactly at N-1
    if steps[-1] != 0:
        steps.append(0)

    total_time_solving = 0.
    numCalls = 0
    for i in steps:
        # first construct what the e-values would be if we were to stop at time i, depending
        # on the value of S_i
        for j in range(n+1):
            # operating under S_{i+1} = j, and that B_{(i+1)} = k, Monte Carlo approx R_{i+1}
            # first we construct the e-values
            # the below variable counts number of calibration weakly below the (i+1)th 
            numCalibBelow = n-j
            # the number of test points weakly below
            numActive = i + 1 - numCalibBelow

            # Rule out impossible cases
            if check_impossible(i,j,n,bh_numCalibAbove,bh_st):
                continue
            
            j_reindexed = j-bh_numCalibAbove

            eValueMultiplier = (n+1.)/(1. + numCalibBelow)
            
            # only need consider case when numActive is nonzero since otherwise, 
            # the value of the program is just 0
            if numActive != 0:
                # at this stage, we randomly sample which of the points are test and
                # which are calibration
                # if the number of calibration weakly below (i+1)th is numCalibBelow
                # then the number of test weakly below is i+1-numCalibBelow
                # if B_{(i+1)} = 1, then we cannot select i
                activeTestInds = np.random.choice(i+1, numActive, replace=False)
                
                # construct the randomly subsetted mcSimilarityMatrix
                mcSimilarityMatrix = dacs.subset(sortedSimilarityMatrix,activeTestInds)



                if numActive == 1:
                    # check self consistency:
                    if eValueMultiplier >= m/alpha:
                        # in this case, just will accept:
                        # (technically, this assumes that all diagonal entries are same)
                        R[i,j_reindexed] = -(0.5*gamma-1)#-(0.5*gamma-1)
                    else:
                        # if self-consistency violated incur 0 reward
                        R[i,j_reindexed] = worst_case_value
                
                # in non-trivial case, we must now use the pgd
                else:
                    # stil check self-consistency
                    # if self-consistency is exact, reject everything
                    if numActive * eValueMultiplier == m/alpha:
                        R[i,j_reindexed]= -(0.5*(gamma)*mcSimilarityMatrix.sum()\
                                        -numActive)
                    
                    # in this case, there is strict feasibility
                    elif numActive * eValueMultiplier > m/alpha:
                        if custom:
                            start = time.time()
                            solver = dacs.MarkowitzPGDSolver(S=mcSimilarityMatrix, C=(alpha/m)*eValueMultiplier, \
                                                                gamma=gamma,abs_tol=default_tol,
                                                                rel_tol=default_tol)
                            xopt = solver.solve()
                            end = time.time()
                            numCalls += 1
                        else:
                            start = time.time()
                            y = cp.Variable(numActive)
                            prob = cp.Problem(cp.Minimize((gamma*0.5*cp.quad_form(y, cp.psd_wrap(mcSimilarityMatrix))) \
                                                        - np.ones(numActive).T @ y),
                                        [0 <= y,
                                        y <= (alpha/m) * eValueMultiplier * (np.ones(numActive).T @ y),
                                        y <= 1
                                        ])
                            prob.solve(verbose=False, solver=cp.MOSEK)
                            xopt = y.value
                            end = time.time()


                        total_time_solving += (end-start)
                        xoptSquared = xopt**2

                        programValue = xopt.sum() - 0.5*gamma*(xopt.T @ mcSimilarityMatrix @ xopt\
                            + np.dot(mcSimilarityMatrix.diagonal(), xopt-xoptSquared))
                            
                            # # construct eValue array
                            # eValues = np.full(numActive,eValueMultiplier)
                            # programValue = markowitz_eValueOptimizer(eValues, mcSimilarityMatrix, alpha, \
                            #                     gamma, m, worst_case_value, expectedValue=True)[1]

                        R[i,j_reindexed] = programValue

                    else:
                        R[i,j_reindexed] = worst_case_value
                
            else:
                R[i,j_reindexed] = worst_case_value # get reward worst_case_value for self-consistency infeasible solution
    return R, total_time_solving


def constructVs(bh_st, # index of BH stopping time
                bh_numCalibAbove, # number of calibration points weakly above at bh_st
                R, # 2d array of (measurable) rewards
                 n, # number of calibration
                 m, # number of test
                 skip=1 # value of skipper to fill out every 
                 # skip^th row of DP table
                 ):
    '''Given rewards (R), construct Vs, via dynamic programming'''

    print("Constructing Snell envelope...")

    poss_num_vals_j = n-bh_numCalibAbove+1
    V = np.full((bh_st+1, poss_num_vals_j), np.nan)
    V[0] = R[0]

    steps = list(range(bh_st, -1, -skip))
    
    # Ensure that N-1 is included, even if the last step is not 
    # exactly at N-1
    if steps[-1] != 0:
        steps.append(0)
    
    # beacuse the reverse DP is on a backwards filtration
    # we proceed in forwards order
    steps = list(reversed(steps))
    for i in tqdm(steps):
         # in this case, V = Y
        if i == 0:
            continue
        
        # perform DP update
        # First we compute E[V_{k+1} | F_k]
        # To do so, we loop over all possible values of S_{k}, B_{-k}
        for j in range(max(bh_numCalibAbove,n-(i+1)), min(n,bh_st+1-(i+1) + bh_numCalibAbove)+1):
            j_reindexed = j-bh_numCalibAbove
            # operating under S_{i+1} = j, compute E[V_{i+1} | F_i]
            # under hypergeometric assumption. Here, F_i contains the information
            # of just (S_{i+1}, B_{i+1}), and we would like to know
            # what is the conditional distribution of (S_i, B_i).
            # the conditional dist of B_i is Bern(#calib below thresh/(i+1))
            # the below variable counts number of calibration strictly below 
            # the (i+1)th 
            
            # The calculations are more complicated in the general
            # skip > 1 hypergeometric case
            numCalibBelow = n-j

            # in general, the distribution will be hypergeometric
            # the joint distribution of (j,k) (i.e., of (S,B)) is 
            bernP = np.array([1.-numCalibBelow/(i+1), numCalibBelow/(i+1)])
            
            # take the conditional expectation
            if j == n:
                # If j == n, then numCalibWeaklyAbove is n
                # then we we know that everything below is 
                # test and so 
                expectation = V[max(i-skip,0)][j_reindexed]
            else:                
                # We condition on the value of B (B is 
                # indicator of calib/test for timestamp i-skip)
                if skip == 1:
                    EV_next_given_V_current_given_0 = V[i-1][j_reindexed]
                    EV_next_given_V_current_given_1 = V[i-1][j_reindexed+1]

                    support = np.array([EV_next_given_V_current_given_0, \
                                        EV_next_given_V_current_given_1])
                    feasible = np.logical_not(np.array([check_impossible(i-1,j,n,bh_numCalibAbove,bh_st), \
                                            check_impossible(i-1,j+1,n,bh_numCalibAbove,bh_st)]))
                    expectation = (bernP[feasible]*support[feasible]).sum()
                else:
                    # possible values of j are j,...,min(j+skip,n)

                    # (take max(i-skip,0) due to potential divisibility issues)
                    support = V[max(i-skip,0),np.arange(j_reindexed,min(j+skip,n)-bh_numCalibAbove+1)]
                    feasible = np.logical_not(vectorized_check_impossible(max(i-skip,0),\
                                                            np.arange(j,min(j+skip,n)+1),\
                                                            n,bh_numCalibAbove,bh_st))
                    pmf = hypergeom.pmf(np.arange(j,min(j+skip,n)+1)-j,i+1,numCalibBelow,\
                                        i-max(i-skip,0))
                    
                    if np.any(feasible):
                        expectation = vectorized_custom_multiply(support[feasible], pmf[feasible]).sum()
                    else:
                        print("Error: nothing feasible")

            V[i,j_reindexed] = max(R[i,j_reindexed], expectation)
    return V