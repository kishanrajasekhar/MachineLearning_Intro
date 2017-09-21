import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
import sys


data = np.genfromtxt("data/X_train.txt")

Xtr = data[:10000] # 10000 points (of 14 features)
Xva = data[10000:20000]

data2 = np.genfromtxt("data/Y_train.txt")
Ytr = data2[:10000]
Yva = data2[10000:20000]
