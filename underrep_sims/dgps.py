import numpy as np


def dgp(n,x_noise,setting):
    dgps = [dgp_random_hypo_discrete1,\
           dgp_random_hypo_discrete2,\
            dgp_random_hypo_discrete3,\
            dgp_random_hypo_discrete4,\
            dgp_random_hypo_discrete5,\
            dgp_random_hypo_discrete6]

    return dgps[setting](n,x_noise)


def dgp_random_hypo_discrete1(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = X + gaussian noise. X's are sampled according to
    a hierarchical Gaussian mixture.
    """
    y_noise=1
    prob = np.full(3,1./3)
    sub_prob1 = np.array([0.8, 0.05, 0.15])
    sub_prob2 = np.array([0.2, 0.6, 0.2])
    sub_prob3 = np.array([0.2, 0.2, 0.6])
    
    means = np.array([-0.5,1.5,2.])
    zs = np.random.choice(3, size=n,p=prob).astype(int)
    noise = x_noise*np.random.normal(size=n)
    Xs = np.where(zs == 0, means[np.random.choice(3,size=n,p=sub_prob1)] + noise,\
                  np.where(zs == 1, means[np.random.choice(3,size=n,p=sub_prob2)] + noise,\
                           means[np.random.choice(3,size=n,p=sub_prob3)] + noise))

    Ys = Xs+y_noise*np.random.normal(size=n)

    return Xs,Ys,zs,3


def dgp_random_hypo_discrete2(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = X^2-1 + gaussian noise. X's are sampled according to
    a hierarchical Gaussian mixture.
    """
    y_noise=1.
    prob = np.array([0.5,0.5])
    sub_prob1 = np.array([0.8, 0.05, 0.15])
    sub_prob2 = np.array([0.15, 0.75, 0.1])
    
    means = np.array([0.,-2.,1.5])
    zs = np.random.choice(2, size=n,p=prob).astype(int)
    noise = x_noise*np.random.normal(size=n)
    Xs = np.where(zs == 0, means[np.random.choice(3,size=n,p=sub_prob1)] + noise,\
                means[np.random.choice(3,size=n,p=sub_prob2)] + noise)
    Ys = (Xs**2)-1.+y_noise*np.random.normal(size=n)

    return Xs,Ys,zs,2


def dgp_random_hypo_discrete3(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = 2*cos(X) + gaussian noise. X's are sampled according to
    a hierarchical Gaussian mixture.
    """
    y_noise = 1.
    prob = np.array([0.5, 0.5])
    sub_prob1 = np.array([0.05,0.85,0.1])
    sub_prob2 = np.array([0.4,0.2,0.4])
    noise = x_noise*np.random.normal(size=n)
    zs = np.random.choice(2, size=n,p=prob).astype(int)
    means = np.array([0,-np.pi,0.7])
    Xs = np.where(zs == 0, means[np.random.choice(3,size=n,p=sub_prob1)] + noise,\
                   means[np.random.choice(3,size=n,p=sub_prob2)] + noise)

    Ys = 2*np.cos(Xs)+y_noise*np.random.normal(size=n)

    return Xs,Ys,zs,2

def dgp_random_hypo_discrete4(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = 3*1(X>0)-1.5 + gaussian noise. X's are sampled 
    according to a hierarchical Gaussian mixture.
    """
    y_noise = 1.
    prob = np.full(4,1./4)
    sub_prob1 = np.array([0.2, 0., 0., 0., 0.8])
    sub_prob2 = np.array([0, 0.22, 0.35, 0.43, 0.])
    sub_prob3 = np.array([0.15, 0.35, 0.15, 0.1, 0.25])
    sub_prob4 = np.array([0.2, 0.05, 0.05, 0.05, 0.65])
    zs = np.random.choice(4, size=n,p=prob).astype(int)
    means = np.array([-2,-1,0.,1.5,3.])
    noise = x_noise*np.random.normal(size=n)
    
    Xs = np.where(zs == 0, means[np.random.choice(5,size=n,p=sub_prob1)] + noise,\
                np.where(zs == 1, means[np.random.choice(5,size=n,p=sub_prob2)] + noise,\
                np.where(zs == 2, means[np.random.choice(5,size=n,p=sub_prob3)] + noise,\
                means[np.random.choice(5,size=n,p=sub_prob4)] + noise)))
    
    Ys = 3*((Xs > 0).astype(int))-1.5+y_noise*np.random.normal(size=n)
    return Xs,Ys,zs,4


def dgp_random_hypo_discrete5(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = X^2-1 + gaussian noise. X's are sampled according to
    a hierarchical Gaussian mixture.
    """
    y_noise=1.
    prob = np.array([0.7,0.3])
    sub_prob1 = np.array([0.8, 0.05, 0.15])
    sub_prob2 = np.array([0.15, 0.75, 0.1])
    
    means = np.array([0.,-2.,1.5])
    zs = np.random.choice(2, size=n,p=prob).astype(int)
    noise = x_noise*np.random.normal(size=n)
    Xs = np.where(zs == 0, means[np.random.choice(3,size=n,p=sub_prob1)] + noise,\
                means[np.random.choice(3,size=n,p=sub_prob2)] + noise)
    Ys = (Xs**2)-1.+y_noise*np.random.normal(size=n)

    return Xs,Ys,zs,2



def dgp_random_hypo_discrete6(n,x_noise):
    """
    This function generates a dataset of X's and Y's. 
    Y = X^3-X + gaussian noise. X's are sampled 
    according to a hierarchical Gaussian mixture.
    """
    y_noise=1.
    prob = np.full(3,1./3)
    sub_prob1 = np.array([1.,0.,0.])
    sub_prob2 = np.array([0.2,0.2,0.6])
    sub_prob3 = np.array([0.2,0.6,0.2])
    zs = np.random.choice(3, size=n,p=prob).astype(int)
    means = np.array([-.75,0.5,1.2])
    noise = x_noise*np.random.normal(size=n)
    Xs = np.where(zs == 0, means[np.random.choice(3,size=n,p=sub_prob1)]+noise,\
                np.where(zs==1, means[np.random.choice(3,size=n,p=sub_prob2)]+noise,\
                    means[np.random.choice(3,size=n,p=sub_prob3)]+noise))
    Ys = Xs**3+Xs+y_noise*np.random.normal(size=n)

    return Xs,Ys,zs,3