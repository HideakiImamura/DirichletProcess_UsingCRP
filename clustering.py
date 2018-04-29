import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from scipy import special
from sklearn.decomposition import PCA
import gibbs1 as g1
import gibbs2 as g2
import collapsed_gibbs as cg
import crp as crp
import time


def make_dataset(n, mus, sigmas):
    num_datas = len(mus)
    xs = []
    for i in range(num_datas):
        xs.append(np.array([np.random.multivariate_normal(mus[i], sigmas[i]) for _ in range(n // num_datas)]))
    return np.random.permutation(np.concatenate(xs))


def make_mu_sigmas_triangle(K):
    mus = []
    sigmas = []
    for i in range(K):
        k = np.floor((np.sqrt(8 * i + 7) + 1) / 2)
        l = i + 1 - k * (k - 1) / 2
        mus.append([10 * (k - l), 10 * (l - 1)])
        sigmas.append([[1, 0], [0, 1]])
    return mus, sigmas


def make_mu_sigmas_square(K):
    mus = []
    sigmas = []
    for i in range(K):
        k = np.ceil((1 + np.sqrt(2 * i + 1)) / 2)
        l = i - 2 * k ** 2 + 6 * k - 4
        print("i = ", i)
        print("k = ", k)
        print("l = ", l)
        if 1 <= l <= k - 1:
            mus.append([10 * (k - l), 10 * (l - 1)])
        elif k <= l <= 2 * k - 2:
            mus.append([10 * (k - l), 10 * (2 * k - l - 1)])
        elif 2 * k - 1 <= l <= 3 * k - 2:
            mus.append([10 * (l - 3 * k + 2), 10 * (2 * k - 1 - l)])
        elif 3 * k - 1 <= l <= 4 * k - 4:
            mus.append([10 * (l - 3 * k + 2), 10 * (l - 4 * k + 3)])
        elif i == 0:
            mus.append([0, 0])
        sigmas.append([[1, 0], [0, 1]])
    return mus, sigmas


def preprocess(train_datas, K):
    pca = PCA(n_components=30)
    xs = pca.fit_transform(np.concatenate([train_datas[i]
                                           for i in range(0, K)], axis=0))
    ans = []
    for i in range(len(xs)):
        ans.append([xs[i], i // 500])
    np.random.shuffle(ans)
    return ans


def show_result(xs, zs, title, K, time):
    print("elapsed_time for %s: %f" % (title, time))
    xss = [np.array([xs[i] for i in range(len(xs)) if zs[i] == k])
           for k in range(K)]
    us = [xss[k][:, 0] for k in range(K)]
    vs = [xss[k][:, 1] for k in range(K)]
    for k in range(K):
        plt.scatter(us[k], vs[k])
    plt.title(title + " elapsed time = " + str(format(time, ".2f")))
    plt.savefig("./results/" + title + ".png")


def num_datas_to_each_class(zs, K):
    ans = np.zeros(K)
    for z in zs:
        try:
            ans[z-1] += 1
        except IndexError:
            print("IndexError")
    return ans


def pseudo_F(xs, zs, K):
    n = len(xs)
    centroids = [np.mean([xs[i] for i in range(n) if zs[i] == k], axis=0) for k in range(K)]
    center = np.mean(xs, axis=0)
    T = np.sum(np.sum((xs - center) ** 2, axis=1))
    W = np.sum([np.sum(np.sum([(xs[i] - centroids[k]) ** 2 for i in range(n) if zs[i] == k])) for k in K])
    return (T - W) * (n - K) / ((K - 1) * W)


def iteration_crp(S, xs, num):
    col = ["Iteration Number", "Elapsed Time",
                        "Estimated Number of Classes", "Number of Datas Distributed to Each Classes"]
    df = pd.DataFrame(columns=col)
    for i in range(num):
        start = time.time()
        zs, K = crp.gibbs(S, xs)
        elapsed = time.time() - start
        num = num_datas_to_each_class(zs, K)
        df = df.append(pd.DataFrame([[i, elapsed, K, num]], columns=col))
        show_result(xs, zs, "CRP iteration %d" % i, K, elapsed)
    print(df)
    df.to_csv("./results/result.csv")


K = 5
n = 1000
# mus, sigmas = make_mu_sigmas_triangle(K)
mus, sigmas = make_mu_sigmas_square(K)
print("mus = ", mus)
xs = make_dataset(n, mus, sigmas)
# xs = xs - xs.mean(axis=0)
x, y = xs[:, 0], xs[:, 1]
plt.scatter(x, y)
plt.scatter(xs.mean(axis=0)[0], xs.mean(axis=0)[1])
#plt.show()

S = 20

iteration_crp(S, xs, 50)
'''
start = time.time()
zs1 = g1.gibbs(S, K, xs)
elapsed_time = time.time() - start
show_result(xs, zs1, "gibbs1", K, elapsed_time)
start = time.time()
zs2 = g2.gibbs(S, K, xs)
elapsed_time = time.time() - start
show_result(xs, zs2, "gibbs2", K, elapsed_time)
start = time.time()
zs3 = cg.gibbs(S, K, xs)
elapsed_time = time.time() - start
show_result(xs, zs3, "ver3", K, elapsed_time)
'''

