import numpy as np

def P_Rplus(x):
    n = x.size
    y = x.copy()
    for i in range(n):
        if y[i] < 0:
            y[i] = 0
    return y


def Constant(x):
    return x
