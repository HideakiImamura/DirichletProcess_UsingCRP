import numpy as np
from scipy import stats


def sample2_z(x, mus, tau, pi):
    a = [stats.multivariate_normal.pdf(x, mus[i], (np.identity(len(x))/tau)) * pi[i]
         for i in range(len(mus))]
    return np.argmax(np.random.multinomial(1, np.asarray(a)/np.sum(a)))


def sample2_mu(k, zs, xs, tau, mu0, rho0):
    n_k = np.sum([int(zs[i] == k) for i in range(len(zs))])
    if n_k == 0:
        x_k = [0 for _ in range(len(xs[0]))]
    else:
        x_k = np.sum([int(zs[i] == k)*(np.asarray(xs[i]))
                      for i in range(len(zs))], axis=0) / n_k
    mu = (n_k*x_k + rho0 * mu0) / (n_k+rho0)
    sigma = np.identity(len(x_k)) / (tau*(n_k+rho0))
    return np.random.multivariate_normal(mu, sigma)


def sample2_tau(zs, xs, mu0, rho0, a0, b0, K):
    a_n = a0 + len(zs)*len(xs[0])
    b_n = b0
    for k in range(K):
        n_k = np.sum([int(zs[i] == k) for i in range(len(zs))])
        if n_k == 0:
            x_k = [0 for _ in range(len(xs[0]))]
        else:
            x_k = np.sum([int(zs[i] == k)*(np.asarray(xs[i]))
                          for i in range(len(zs))], axis=0) / n_k
        b_n += np.sum([int(zs[i] == k)*np.sum((xs[i]-x_k)**2)
                       for i in range(len(zs))])/2
        b_n += n_k * rho0 * np.sum((x_k - np.asarray(mu0))**2) / (2*(rho0+n_k))
    return np.random.gamma(a_n, 1/b_n)


def sample2_pi(zs, alpha, K):
    n_ks = [np.sum([int(zs[i] == k) for i in range(len(zs))])
            for k in range(K)]
    alpha_ks = [alpha + n_ks[k] for k in range(K)]
    return np.random.dirichlet(alpha_ks)



def joint_log_proba2(xs, zs, mus, tau, pis, mu0, rho0, a0, b0, alpha):
    ans = 0
    n = len(xs)
    d = len(xs[0])
    K = len(mus)
    for i in range(len(zs)):
        ans += stats.multivariate_normal.logpdf(xs[i], mus[zs[i]], np.identity(d)/tau)
        ans += np.log(pis[zs[i]])
    for k in range(K):
        ans += stats.multivariate_normal.pdf(mus[k], mu0, np.identity(d)/(rho0*tau))
    ans += stats.gamma.logpdf(tau, a0, scale=1/b0)
    ans += stats.dirichlet.logpdf(pis, [alpha for _ in range(len(pis))])
    return ans


def gibbs(S, K, xs):
    d = len(xs[0])
    n = len(xs)
    alpha = 0.5
    mu0 = [0 for _ in range(d)]
    rho0 = 1
    a0 = 1
    b0 = 1
    mus = [np.random.uniform(size=d) for _ in range(K)]
    pis = [1/K for _ in range(K)]
    tau = 1
    zs = [[0 for _ in range(n)] for _ in range(S)]
    probas = []
    for s in range(0, S):
        for i in range(n):
            zs[s][i] = sample2_z(xs[i], mus, tau, pis)
        for k in range(K):
            mus[k] = sample2_mu(k, zs[s], xs, tau, mu0, rho0)
        tau = sample2_tau(zs[s], xs, mu0, rho0, a0, b0, K)
        pis = sample2_pi(zs[s], alpha, K)
        probas.append(joint_log_proba2(xs, zs[s], mus, tau, pis, mu0, rho0, a0, b0, alpha))
    return zs[np.argmax(probas)]




