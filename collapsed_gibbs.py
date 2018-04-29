import numpy as np
import mydist as dist
import scipy.special as sp
import time


def sample(i, K, xs, zs, mu0, rho0, a0, b0, alpha):
    n = len(xs)
    d = len(xs[0])
    print("n_kis calculation start")
    start = time.time()
    n_kis = [np.sum([int(zs[j] == k)+1 for j in range(n) if j != i]) for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    n_kis_plus_alpha = (np.asarray(n_kis) + alpha).tolist()
    print("x_kis calculation start")
    start = time.time()
    x_kis = [np.sum([int(zs[j] == k) * np.asarray(xs[j])
                     for j in range(n) if j != i], axis=0) / n_kis[k] for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("m_kis calculation start")
    start = time.time()
    m_kis = [(n_kis[k]*x_kis[k]+rho0*np.asarray(mu0))/(n_kis[k]+rho0) for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("a_kis calculation start")
    start = time.time()
    a_kis = [a0 + n_kis[k]*d/2 for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("b_kis calculation start")
    start = time.time()
    b_kis = [b0 + np.sum([int(zs[j] == k)*np.sum((xs[j]-x_kis[k])**2) for j in range(n) if j != i])
             + n_kis[k]*rho0*np.sum((x_kis[k] - np.asarray(mu0))**2) / (2*(rho0+n_kis[k])) for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("probas calculation start")
    start = time.time()
    probas = [dist.student_t(xs[i], m_kis[k], 2*a_kis[k], (1+1/(n_kis[k]+rho0))*2*b_kis[k])*n_kis_plus_alpha[k]
              for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    return np.argmax(np.random.multinomial(1, np.asarray(probas)/np.sum(probas)))


def joint_log_proba(K, xs, zs, mu0, rho0, a0, b0, alpha):
    ans = 0
    n = len(xs)
    d = len(xs[0])
    print("n_ks calculation start")
    start = time.time()
    n_ks = [np.sum([int(zs[i] == k)+1 for i in range(n)]) for k in range(K)]
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("x_ calculation start")
    start = time.time()
    x_ = np.sum(xs, axis=0) / n
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("bn calculation start")
    start = time.time()
    bn = b0 + np.sum((xs - x_)**2)/2 + n*rho0*np.sum((x_ - mu0)**2)/(2*(rho0+n))
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    print("ans calculation start")
    start = time.time()
    ans += -np.log(1+(n/rho0))/2 + a0*np.log(b0/bn) - n*d*np.log(bn)/2 + \
           sp.gammaln(a0+n*d/2) - sp.gammaln(a0) + sp.gammaln(K*alpha) \
           - sp.gammaln(n+K*alpha) - K*sp.gammaln(alpha)
    for k in range(K):
        ans += -n_ks[k]*d*np.log(2*np.pi)/2 + sp.gammaln(n_ks[k]+alpha)
    elapsed = time.time() - start
    print("elapsed time : %f" % elapsed)
    return ans


def gibbs(S, K, xs):
    n = len(xs)
    d = len(xs[0])
    m = min(xs.min(0))
    M = max(xs.max(0))
    mu0 = np.random.uniform(low=m, high=M, size=d).tolist()
    rho0 = 1
    a0 = 1
    b0 = 1
    alpha = 0.5
    zs = [[0 for _ in range(n)] for _ in range(S)]
    probas = []
    for s in range(S):
        print("Iteration %d start" % (s+1))
        for i in range(n):
            zs[s][i] = sample(i, K, xs, zs[s], mu0, rho0, a0, b0, alpha)
        probas.append(joint_log_proba(K, xs, zs[s], mu0, rho0, a0, b0, alpha))
    return zs[np.argmax(probas)]