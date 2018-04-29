import numpy as np
from scipy import stats


def sample1_z(x, mus):
    multi_normal = stats.multivariate_normal
    a = [multi_normal.pdf(x, mus[i], np.identity(len(x)))
         for i in range(0, len(mus))]
    return np.argmax(np.random.multinomial(1, np.asarray(a)/np.sum(a)))


def sample1_mu(k, zs, xs, mu0, sigma2):
    n_k = np.sum([int(zs[i] == k) for i in range(len(zs))])
    if n_k == 0:
        x_k = [0 for _ in range(len(xs[0]))]
    else:
        x_k = np.sum([int(zs[i] == k)*(np.asarray(xs[i]))
                      for i in range(len(zs))], axis=0) / n_k
    mu = (n_k*x_k + sigma2*mu0) / (n_k+sigma2)
    return np.random.multivariate_normal\
        (mu, np.identity(len(x_k))/(n_k+sigma2))


def joint_log_proba(xs, zs, mus, mu0, sigma2, pi0):
    ans = 1
    n = len(xs)
    d = len(xs[0])
    K = len(mus)
    multi_normal = stats.multivariate_normal
    for i in range(n):
        ans += (multi_normal.logpdf(xs[i], mus[zs[i]], np.identity(d))
                + np.log(pi0[zs[i]]))
    for k in range(K):
        ans += multi_normal.logpdf(mus[k], mu0, sigma2*np.identity(d).tolist())
    return ans


def gibbs(S, K, xs):
    d = len(xs[0])
    n = len(xs)
    M = max(xs.max(0))
    m = min(xs.min(0))
    mus = [np.random.uniform(low=m, high=M, size=d) for _ in range(K)]
    #mus = [[0 for _ in range(d)] for _ in range(K)]
    #mu0 = [0 for _ in range(d)]
    mu0 = np.random.uniform(low=m, high=M, size=d).tolist()
    sigma2 = 1
    pi0 = [1/K for _ in range(K)]
    zs = [[0 for _ in range(n)] for _ in range(S)]
    probas = []
    for s in range(0, S):
        #print("s = %d start" % s)
        for i in range(n):
            zs[s][i] = sample1_z(xs[i], mus)
        for k in range(K):
            #print("k = %d start" % k)
            mus[k] = sample1_mu(k, zs[s], xs, mu0, sigma2)
        probas.append(joint_log_proba(xs, zs[s], mus, mu0, sigma2, pi0))
    max_s = np.argmax(probas)
    return zs[max_s]