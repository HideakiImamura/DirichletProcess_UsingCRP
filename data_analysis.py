import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


df = pd.read_csv('./results/result.csv')
plt.hist(df["Estimated Number of Classes"])
plt.savefig("./results/classes.png")
plt.clf()

df2 = df["Number of Datas Distributed to Each Classes"]
a = [re.split('\s|\.\s*', df2[i]) for i in range(len(df2))]
for l in a:
    l.remove('[')
    l.remove(']')
    try:
        while True:
            l.remove('')
    except ValueError:
        pass
    print(l)
print(a)
a = [np.sort(list(map(int, a[i])))[::-1] for i in range(len(df2))]
a.sort(key=len)
pd.DataFrame(a).to_csv("./results/dist_class_num.csv")
