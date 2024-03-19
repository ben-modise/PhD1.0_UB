import numpy as np
import functools


#def counter(F):
#    @functools.wraps(F)
#    def wrapper_function(*args, **kwargs):
#        wrapper_function.evals += 1
#        return F(*args, **kwargs)
#    wrapper_function.called = 0
#    wrapper_function.__name__ = F.__name__
#    return wrapper_function

def modified_exponential(x):
    '''Problem 1 from Abubakar et. al/ Results in Applied Mathematics
     4 (2019) 10069'''
    n = x.size
    v = np.concatenate(([[0]],x[1:]))
    y = np.exp(x) + v - np.ones([n,1])
    return y

def logarithmic(x):
    '''Problem 2 from Abubakar et. al/ Results in Applied Mathematics
    4 (2019) 10069'''
    n = x.size
    y = np.log(np.abs(x)+1) - x/n
    return y
    
def logarithmic2(x):
    n = x.size
    y = np.log(x+1) - x/n
    return y

def problem3(x):
    y = 2*x - np.sin(abs(x))
    return y

def strictly_convex(x):
    y = np.exp(x) - 1
    return y

def linear_monotone(x):
    #y = np.zeros([x.size,1])
    y0 = 2.5*x[0] + x[1] -1
    yn = x[-2] + 2.5*x[-1] -1
    y_inner = x[:-2] + x[2:] + 2.5*x[1:-1] - 1
    y = np.concatenate(([y0], y_inner, [yn]))
    return y

def tridiagonal_exponential(x):
    n = x.size
    h = 1/(n+1)
    y = np.zeros([n,1])
    y[0] = x[0] - np.exp(np.cos(h*(x[0] + x[1])))
    y[-1] = x[-1] - np.exp(np.cos(h*(x[-1] + x[-2])))
    for i in range(1,n-1):
        y[i] = x[i] -  np.exp(np.cos(h*(x[i-1] + x[i])))
    return y

def tridiagonal(x):
    n = x.size
    y0 = x[0]*(x[0]**2 +x[1]**2)-1
    yn = x[n-1]*(x[n-2]**2 + x[n-1]**2) -1
    x_up = x[:n-2]
    x_middle = x[1:n-1]
    x_down = x[2:]
    y_middle_right = x_up**2 + 2*x_middle**2 + x_down**2
    y_middle = x_middle*y_middle_right - 1
    y = np.concatenate(([y0], y_middle, [yn]))
    return y

def tridiagonal_via_loop(x):
    n = x.size
    y = np.zeros([n,1])
    y[0] = x[0]*(x[0]**2 + x[1]**2) -1
    y[-1] = x[-1]*(x[-1]**2 + x[-2]**2) -1
    for i in range(1,n-1):
        y[i] = x[i]*(x[i-1]**2 +2*x[i]**2 + x[i+1]**2) - 1
    return y

def problem7(x):
    n = x.size
    y = np.zeros([n,1])
    y[0] = 2*x[0] - x[1] + np.exp(x[0]) -1
    y[-1] = 2*x[-1] - x[-2] + np.exp(x[-1]) -1
    y[1:-1] = 2*x[1:-1] - x[:-2] - x[2:] + np.exp(x[1:-1]) - 1
    return y

def problem8(x):
    y = 2*x - np.sin(x)
    return y

def problem9(x):
    y = np.exp(x) + x -1
    return y
