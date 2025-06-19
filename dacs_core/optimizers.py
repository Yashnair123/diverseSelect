import numpy as np
import cvxpy as cp
import time


def sharpe_eValueOptimizer(eValues, similarityMatrix, alpha, m, worst_case_value, expectedValue, bernMC=50):
    d = len(eValues)
    y = cp.Variable(d)
    
    # Return 0 and 0 in cases wherein the program is infeasible

    # First case is if there are non-nonzero e-values
    if np.count_nonzero(eValues) == 0:
        return np.zeros(d), worst_case_value
    
    # Second case is if the value of the e-values are so small
    # that rejecting all the non-zero e-values is impossible 
    # (hence, no rejection set will satisfy self-consistency)
    if np.max(eValues) < m/(alpha*np.count_nonzero(eValues)):
        return np.zeros(d), worst_case_value
    start = time.time()
    prob = cp.Problem(cp.Minimize((cp.quad_form(y, cp.psd_wrap(similarityMatrix)))),
                [0 <= y,
                y <= (alpha/m)*eValues, #(self-consistency constraint simplifies 
                # under assumption that 1^\top y = 1)
                np.ones(d).T @ y == 1
                ])
    prob.solve(verbose=False, solver=cp.MOSEK)
    end = time.time()
    x = y.value * (1./np.max(y.value)) # scale up so that largest entry is a 1
    if expectedValue:
        avgVal = 0.
        for _ in range(bernMC):
            b = np.random.binomial(1,p=np.clip(x,0.,1.))
            if np.sum(b) > 0:
                avgVal += (np.ones(d).T @ b/np.sqrt(b.T @ similarityMatrix @ b))/bernMC
            else:
                avgVal += 0.
        b = np.random.binomial(1,p=np.clip(x,0.,1.))
        # avgVal = (np.ones(d).T @ x/np.sqrt(x.T @ similarityMatrix @ x))
        return b, avgVal
    else:
        #print(similarityMatrix)
        # print(x)

        b = np.random.binomial(1,p=np.clip(x,0.,1.))
        if np.sum(b) > 0:
            return b, np.ones(d).T @ b/np.sqrt(b.T @ similarityMatrix @ b)
        else:
            return b, worst_case_value


def sharpe_eValueOptimizer_returnx(eValues, similarityMatrix, alpha, m):
    d = len(eValues)
    y = cp.Variable(d)
    
    # Return 0 and 0 in cases wherein the program is infeasible

    # First case is if there are non-nonzero e-values
    if np.count_nonzero(eValues) == 0:
        return np.zeros(d)
    
    # Second case is if the value of the e-values are so small
    # that rejecting all the non-zero e-values is impossible 
    # (hence, no rejection set will satisfy self-consistency)
    if np.max(eValues) < m/(alpha*np.count_nonzero(eValues)):
        return np.zeros(d)
    start = time.time()
    prob = cp.Problem(cp.Minimize((cp.quad_form(y, cp.psd_wrap(similarityMatrix)))),
                [0 <= y,
                y <= (alpha/m)*eValues, #(self-consistency constraint simplifies 
                # under assumption that 1^\top y = 1)
                np.ones(d).T @ y == 1
                ])
    prob.solve(verbose=False, solver=cp.MOSEK)
    end = time.time()
    x = y.value * (1./np.max(y.value)) # scale up so that largest entry is a 1
    
    return x


def markowitz_eValueOptimizer(eValues, similarityMatrix, alpha, gamma, m, worst_case_value, expectedValue):
    # worst_case_value = 0.
    d = len(eValues)
    y = cp.Variable(d)
    
    # Return 0 and 0 in cases wherein the program is infeasible

    # First case is if there are non-nonzero e-values
    if np.count_nonzero(eValues) == 0:
        return np.zeros(d), worst_case_value
    
    # Second case is if the value of the e-values are so small
    # that rejecting all the non-zero e-values is impossible 
    # (hence, no rejection set will satisfy self-consistency)
    if np.max(eValues) < m/(alpha*np.count_nonzero(eValues)):
        return np.zeros(d), worst_case_value
    
    prob = cp.Problem(cp.Minimize((gamma*0.5*cp.quad_form(y, cp.psd_wrap(similarityMatrix))) \
                                  - np.ones(d).T @ y),
                [0 <= y,
                y <= (alpha/m) * eValues * (np.ones(d).T @ y),
                y <= 1
                ])
    prob.solve(verbose=False, solver=cp.MOSEK)
    # print(y.value)
    x = np.clip(y.value, 0., 1.) # scale up so that largest entry is a 1
    b = np.random.binomial(1,p=np.clip(x,0.,1.)) # randomize rejections
    if expectedValue:
        # the expected value of b^T Sigma b is diag(Sigma)^T x + x^ToffDiag(Sigma)x
        # where diag(Sigma) is the vector of diagonal entries of Sigma and offDiag(Sigma)
        # is the matrix Sigma with diagonal set to 0. The above expression simplifies to
        # (defining Diag(Sigma) to be the matrix of zeros except whose diagonal agrees with Sigma)
        # diag(Sigma)^T x + x^T(Sigma - Diag(Sigma))x
        # = diag(Sigma)^T x + x^TSigma x- x^TDiag(Sigma) x
        # = diag(Sigma)^T x + x^TSigma x - diag(Sigma)^T x \odot x
        # where \odot is elementwise product
        xSquared = x**2

        # I place negative sign on the value because the OST assumes you are trying to maximize something
        #avg_val = -float(0.5*gamma*x.T @ similarityMatrix @ x - x.sum())
        #return b, avg_val
        return b, -(0.5 * gamma*(np.dot(similarityMatrix.diagonal(), x-xSquared) + \
                             x.T @ similarityMatrix @ x) - \
                        np.ones(d).T @ x)
    else:
        # I place negative sign on the value because the OST assumes you are trying to maximize something

        return b, -float(0.5*gamma*x.T @ similarityMatrix @ x - x.sum())


def markowitz_eValueOptimizer_returnx(eValues, similarityMatrix, alpha, gamma, m, worst_case_value):
    # worst_case_value = 0.
    d = len(eValues)
    y = cp.Variable(d)
    
    # Return 0 and 0 in cases wherein the program is infeasible

    # First case is if there are non-nonzero e-values
    if np.count_nonzero(eValues) == 0:
        return np.zeros(d), worst_case_value
    
    # Second case is if the value of the e-values are so small
    # that rejecting all the non-zero e-values is impossible 
    # (hence, no rejection set will satisfy self-consistency)
    if np.max(eValues) < m/(alpha*np.count_nonzero(eValues)):
        return np.zeros(d), worst_case_value
    
    prob = cp.Problem(cp.Minimize((gamma*0.5*cp.quad_form(y, cp.psd_wrap(similarityMatrix))) \
                                  - np.ones(d).T @ y),
                [0 <= y,
                y <= (alpha/m) * eValues * (np.ones(d).T @ y),
                y <= 1
                ])
    prob.solve(verbose=False, solver=cp.MOSEK)
    # print(y.value)
    x = np.clip(y.value, 0., 1.) # scale up so that largest entry is a 1
    return x