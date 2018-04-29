import numpy as np
import csv
import itertools
from sklearn.decomposition import PCA


def data_getter(name):
    datas = [[] for _ in range(0, 10)]
    for i in range(0, 10):
        with open('./digit/digit_'+name+str(i)+'.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                #datas[i].append([[float(d) for d in row], i])
                datas[i].append([float(d) for d in row])
    return datas


def gaussian(x, mu, sigma):
    d = len(x)
    xx = np.asarray(x).reshape(d, 1)
    mmu = np.asarray(mu).reshape(d, 1)
    #ssigma = np.matrix(sigma)
    ssigma = sigma[0][0]
    #return (np.exp(-(xx-mmu).T.dot(ssigma.I).dot(xx-mmu)/2) / \
    # ((np.sqrt(2*np.pi)**d)*np.sqrt(np.linalg.det(ssigma)))).tolist()[0][0]
    return (np.exp(-(xx-mmu).T.dot(xx-mmu)/(2*ssigma)) / \
            (np.sqrt(2*np.pi*ssigma*ssigma)**d)).tolist()[0][0]


def multi(x, pi):
    return pi[x]


def sample_z(x, mus):
    a = [gaussian(x, mus[i], np.identity(len(x)).tolist())
         for i in range(0, len(mus))]
    return np.argmax(np.random.multinomial(1, np.asarray(a)/np.sum(a)))


def sample_mu(k, zs, xs, mu0, sigma2):
    n_k = np.sum([int(zs[i] == k) for i in range(0, len(zs))])
    if n_k == 0:
        x_k = [0 for _ in range(len(xs[0][0]))]
    else:
        x_k = np.sum([int(zs[i] == k)*(np.asarray(xs[i][0]))
                      for i in range(0, len(zs))], axis=0) / n_k
    mu = (n_k*x_k + sigma2*mu0) / (n_k+sigma2)
    return np.random.multivariate_normal\
        (mu, np.identity(len(x_k))/(n_k+sigma2))


def joint_log_proba(xs, zs, mus, mu0, sigma2, pi0):
    ans = 1
    n = len(xs)
    d = len(xs[0][0])
    K = len(mus)
    for i in range(0, n):
        ans += (np.log(gaussian(xs[i][0], mus[zs[i]], np.identity(d).tolist()))
                + np.log(multi(zs[i], pi0)))
    for k in range(0, K):
        ans += np.log(gaussian(mus[k], mu0, sigma2*np.identity(d).tolist()))
    return ans


def miss_num(zs, xs, K):
    a = []
    for p in itertools.permutations(range(K)):
        b = 0
        for i in range(0, len(xs)):
            b += (p[zs[i]] != xs[i][1])
        a.append(b)
    print("a = ", a)
    return min(a)


def preprocess(train_datas, K):
    pca = PCA(n_components=30)
    xs = pca.fit_transform(np.concatenate([train_datas[i]
                                           for i in range(0, K)], axis=0))
    ans = []
    for i in range(len(xs)):
        ans.append([xs[i], i//500])
    np.random.shuffle(ans)
    return ans

K = 3
train_datas = data_getter("train")
xs = preprocess(train_datas, K)

S = 10
d = len(xs[0][0])
n = len(xs)
mus = [np.random.uniform(size=d) for _ in range(0, K)]
#mus = [[0 for _ in range(d)] for _ in range(K)]
mu0 = [0 for _ in range(0, d)]
sigma2 = 1
pi0 = [1/K for _ in range(0, K)]
zs = [[0 for _ in range(n)] for _ in range(S)]
probas = []

for s in range(0, S):
    print("s = %d start" % s)
    for i in range(0, n):
        zs[s][i] = sample_z(xs[i][0], mus)
    for k in range(0, K):
        print("k = %d start" % k)
        mus[k] = sample_mu(k, zs[s], xs, mu0, sigma2)
    probas.append(joint_log_proba(xs, zs[s], mus, mu0, sigma2, pi0))
max_s = np.argmax(probas)

miss = miss_num(zs[max_s], xs, K)
print("miss = ", miss)
print("acc = ", (1 - miss/len(xs))*100, "%")
print(max_s)
print(probas)
