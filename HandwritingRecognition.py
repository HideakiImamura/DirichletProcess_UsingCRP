import numpy as np
import pandas as pd
import csv


def data_getter(name):
    datas = [[] for _ in range(0, 10)]
    for i in range(0, 10):
        with open('./digit/digit_'+name+str(i)+'.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                datas[i].append([float(d) for d in row])
    return datas


def estimated_mu_ml(y, n, x):
    return np.sum(x[y], axis=0) / n[y]


def estimated_sigma_ml(y, n, x):
    mu = estimated_mu_ml(y, n, x)
    d = len(mu)
    sigma = np.zeros((d, d))
    for i in range(0, len(x[y])):
        xx = np.asarray(x[y][i])
        sigma += np.matmul((xx-mu).reshape(d, 1), (xx-mu).reshape(1, d))
    return sigma / n[y]


def posterior(x, mu, sigma, n):
    d = len(x)
    xx = np.asarray(x)
    s = np.matrix(sigma)
    a = (- ((xx-mu).dot((s+1e-1*np.identity(d)).I))
         .dot(xx.reshape(d,1)-mu.reshape(d,1))/0.5).tolist()[0][0]
    b = - np.log(np.abs(np.linalg.det(s+1e-1*np.identity(d))))/0.5
    c = np.log(n)
    return a+b+c


def predict(test_data, mus, sigmas, ns):
    ans = [0 for _ in range(0, 10)]
    for x in test_data:
        a = [posterior(x, mus[i], sigmas[i], ns[i]) for i in range(0, 10)]
        k = np.argmax(a)
        ans[k] += 1
    return ans


train_datas = data_getter('train')
test_datas = data_getter('test')
ns = [len(train_datas[i]) for i in range(0, 10)]
num_tests = np.sum([len(test_datas[i]) for i in range(0, 10)])

mus = [estimated_mu_ml(i, ns, train_datas) for i in range(0, 10)]
sigmas = [estimated_sigma_ml(i, ns, train_datas) for i in range(0, 10)]
results = [predict(test_datas[i], mus, sigmas, ns) for i in range(0, 10)]
print(pd.DataFrame(results))
acc = np.sum([results[i][i] for i in range(0, 10)]) / np.sum(num_tests)
print("accuracy = ", acc * 100, "%")

