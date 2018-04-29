import numpy as np
from scipy import special


# class MyDist():
def gaussian(x, mu, sigma):
    d = len(x)
    xx = np.asarray(x).reshape(d, 1)
    mmu = np.asarray(mu).reshape(d, 1)
    # ssigma = np.matrix(sigma)
    ssigma = sigma[0][0]
    # return (np.exp(-(xx-mmu).T.dot(ssigma.I).dot(xx-mmu)/2) / \
    # ((np.sqrt(2*np.pi)**d)*np.sqrt(np.linalg.det(ssigma)))).tolist()[0][0]
    return (np.exp(-(xx - mmu).T.dot(xx - mmu) / (2 * ssigma)) / \
            (np.sqrt(2 * np.pi * ssigma * ssigma) ** d)).tolist()[0][0]


def multi(x, pi):
    return pi[x]


def gamma(x, a, b):
    return (b ** a) * (x ** (a - 1)) * np.exp(-b * x) / special.gamma(a)


def dirichlet(x, a):
    K = len(x)
    xx = np.asarray(x)
    return special.gamma(K * a) * np.prod(xx ** (a - 1)) / (special.gamma(a) ** K)


def student_t(x, mu, a, b):
    d = len(x)
    return special.gamma((a+d)/2) * ((1 + np.sum((np.asarray(x) - np.asarray(mu))**2)/b)**(-(a+d)/2)) / \
           (special.gamma(a/2) * (np.sqrt(b*np.pi)**d))
