import numpy as np
from numpy.linalg import norm
import functools
import pdb

def inner(vector1, vector2):
    return np.sum(vector1*vector2)

def counter(F):
    def wrapper_function(*args, **kwargs):
        wrapper_function.evals += 1
        return F(*args, **kwargs)
    wrapper_function.evals = 0
    wrapper_function.__name__ = F.__name__
    return wrapper_function
    
def line_search(F, xk, dk, beta=1, rho=0.6, sigma=0.01):
    sigma_norm_dk_2 = sigma*np.sum(dk*dk)
    alpha = beta
    w = xk + alpha*dk
    F_w = F(w)
    mk = 0
    while -np.sum(F_w*dk) < alpha*sigma_norm_dk_2:
        alpha *= rho
        w = xk + alpha*dk
        F_w = F(w)
        mk += 1
        if mk >= 1000:
            raise Exception("line search trials exceeded")	  
    return alpha, w, F_w

def zeta(Fw, xk, wk):
    '''Gives the quantity to project xk via xk - zeta*F(zk)
    onto the hyperplane {x in Rn| F(zk)^T(x - z_k) = 0}'''
    norm_Fw_squared = np.sum(Fw*Fw)
    zeta_k = np.sum(Fw*(xk - wk))/norm_Fw_squared
    return zeta_k

def first_iteration(F, F0, x0, line_search=line_search):
    d0 = -F0
    alpha, zk, Fzk = line_search(F, x0, d0)
    x1 = x0 - zeta(F0,x0,zk)*Fzk
    return x1, zk, Fzk

def compute_dk_dmbhp(Fk1, Fk, dk, sk):
    dk1 = -Fk1
    gamma = Fk1 - Fk
    tao_hat = -(inner(gamma, sk))/(norm(sk)**2)
    tao = 1 + max(0, tao_hat)
    assert tao >= 1, f"tao value incorrect"
    y = gamma + tao*sk
    sk_Fk1 = inner(sk, Fk1)
    y_sk = inner(y, sk)
    y_Fk1 = inner(y, Fk1)
    eta_hat = sk_Fk1/y_sk
    eta = 0
    if (sk_Fk1)*(y_Fk1) < 0:
        eta = eta_hat
    varphi_hat = ((1+norm(y)**2/y_sk)*(sk_Fk1/y_sk) - (y_Fk1/y_sk))
    varphi = 0
    if -varphi_hat*sk_Fk1 < 0:
        varphi = varphi_hat
    dk1 = dk1 - varphi*sk + eta*y
    #pdb.set_trace()

    return dk1

def DMBHP(F, x0, Tol=1e-6):
    k = -1
    max_iters = 10000
    F = counter(F)
    Fk = F(x0)
    norm_Fk = norm(Fk)
    if norm_Fk <= Tol:
        return x0, k, 0, norm_Fk
    xk, zk, F_zk = first_iteration(F, Fk, x0)
    sk = zk - x0
    k+=1
    Fk1 = F(xk)
    #norm_Fk = norm(Fk1)
    dk = -Fk
    while norm(Fk1) > Tol:
        k+=1
        dk1 = compute_dk_dmbhp(Fk1, Fk, dk, sk)
        assert inner(dk1, Fk1) <= -norm(Fk1)**2, f"dk nt descent direction"
        alpha_k, zk, Fzk = line_search(F, xk, dk1)
        sk = zk - xk
        if norm(Fzk) < Tol:
            xk = zk
            Fk1 = F(xk)
            break
        xk = xk - zeta(Fzk, xk, zk)*Fzk
        Fk = Fk1
        Fk1 = F(xk)
    return xk,k, F.evals, norm(Fk1)
   

if __name__=="__main__":
    x0 = 2*np.ones([20000,1])
    from problems import *
    F = problem7
    x_star, iters, Feval, norm_F = DMBHP(F,x0)

