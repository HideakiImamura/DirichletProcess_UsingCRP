from scipy import stats
from scipy import special
import numpy as np
import time


def get_class_list(zs, i, class_list, mus):
    num_zi = np.sum([int(zs[j] == zs[i]) for j in range(len(zs))])
    if num_zi >= 2:
        #print("num_zi >= 2")
        return zs, class_list, mus
    else:
        if zs[i] < 0:
            return zs, class_list, mus
        '''print("------------------in get_class_list-------------------------")
        print("class_list = ", class_list)
        print("len(mus) = ", len(mus))
        print("zs[i] = ", zs[i])'''
        new_class_list = list(range(len(class_list)-1))
        new_zs = [zs[j]-1 if zs[j]>zs[i] else zs[j] for j in range(len(zs))]
        mus.pop(zs[i])
        '''print("new_class_list = ", class_list)
        print("new_zs = ", set(new_zs))
        print("len(new_mus) = ", len(mus))'''
        return new_zs, new_class_list, mus


def sample_z(n, i, x, zs, mus, tau, mu0, rho0, alpha, class_list):
    #print("zs = ", set(zs))
    new_zs, new_class_list, new_mus = get_class_list(zs, i, class_list, mus)
    #print("before = ", set(new_zs))
    n_kis = [np.sum([int(new_zs[j] == k) for j in range(len(new_zs)) if j != i]) for k in range(len(new_class_list))]
    a = [stats.multivariate_normal.pdf(x, new_mus[k], (np.identity(len(x))/tau)) * n_kis[k] / (n-1+alpha)
         for k in range(len(new_class_list))]
    mu = np.random.multivariate_normal(mu0, np.identity(len(mu0))/(tau*rho0))
    a.append(stats.multivariate_normal.pdf(x, mu, np.identity(len(x))/tau) * alpha / (n-1+alpha))
    ret = np.argmax(np.random.multinomial(1, np.asarray(a)/np.sum(a)))
    new_zs[i] = ret
    if ret == len(a)-1:
        new_class_list.append(ret)
        new_mus.append(mu)
    if len(set(new_zs) - {-1}) > len(new_mus):
        print("-----------------in sample_z--------------------")
        print("new_class_list = ", new_class_list)
        print("new_zs = ", set(new_zs))
        print("ret = ", ret)
        print("zs[i] = ", zs[i])
        print("len(new_mus) = ", len(new_mus))
        raise ValueError
    #print("new_zs = ", set(new_zs))
    return new_class_list, new_zs, new_mus


def sample_mu(k, zs, xs, tau, mu0, rho0):
    n_k = np.sum([int(zs[i] == k) for i in range(len(zs))])
    if n_k == 0:
        x_k = [0 for _ in range(len(xs[0]))]
    else:
        x_k = np.sum([int(zs[i] == k)*(np.asarray(xs[i]))
                      for i in range(len(zs))], axis=0) / n_k
    mu = (n_k*x_k + rho0 * mu0) / (n_k+rho0)
    sigma = np.identity(len(x_k)) / (tau*(n_k+rho0))
    return np.random.multivariate_normal(mu, sigma)


def sample_tau(zs, xs, mu0, rho0, a0, b0, K):
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


def joint_log_proba(xs, zs, mus, tau, mu0, rho0, a0, b0, alpha):
    ans = 0
    n = len(xs)
    d = len(xs[0])
    K = len(mus)
    print(len(xs))
    print(set(zs))
    print(len(mus))
    for i in range(len(zs)):
        ans += stats.multivariate_normal.logpdf(xs[i], mus[zs[i]], np.identity(d)/tau)
    for k in range(K):
        ans += stats.multivariate_normal.pdf(mus[k], mu0, np.identity(d)/(rho0*tau))
    ans += stats.gamma.logpdf(tau, a0, scale=1/b0)
    n_ks = [np.sum([int(zs[i] == k) for i in range(n)]) for k in range(K)]
    ans += special.gammaln(alpha) - special.gammaln(n+alpha) + K*np.log(alpha)
    for k in range(K):
        for i in range(n_ks[k]):
            ans += np.log(i+1)
    return ans


def gibbs(S, xs):
    d = len(xs[0])
    n = len(xs)
    alpha = 0.01
    mu0 = [0 for _ in range(d)]
    rho0 = 1
    a0 = 1
    b0 = 1
    mus = []
    tau = 1
    class_list = []
    zs = [[-1 for _ in range(n)] for _ in range(S+1)]
    probas = []
    for s in range(0, S):
        print("Iteration %d start" % (s+1))
        for i in range(n):
            class_list, zs[s], mus = sample_z(n, i, xs[i], zs[s], mus, tau, mu0, rho0, alpha, class_list)
        zs[s+1] = zs[s]
        K = int(len(class_list))
        for k in range(K):
            mus[k] = sample_mu(k, zs[s+1], xs, tau, mu0, rho0)
        tau = sample_tau(zs[s+1], xs, mu0, rho0, a0, b0, K)
        probas.append(joint_log_proba(xs, zs[s+1], mus, tau, mu0, rho0, a0, b0, alpha))
        print("class number = %d" % K)
    return zs[np.argmax(probas)], K

