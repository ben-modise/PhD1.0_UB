import numpy as np
from numpy.linalg import norm
import functools

def inner(vector1, vector2):
    return np.sum(vector1*vector2)

def counter(F):
    def wrapper_function(*args, **kwargs):
        wrapper_function.evals += 1
        return F(*args, **kwargs)
    wrapper_function.evals = 0
    wrapper_function.__name__ = F.__name__
    return wrapper_function

def line_search(F, xk, dk, beta=1, rho=0.6, sigma=1e-4):
    sigma_norm_dk_2 = sigma*np.sum(dk*dk)
    alpha = beta
    w = xk + alpha*dk
    F_w = F(w)
    mk = 1
    while -np.sum(F_w*dk) < alpha*sigma_norm_dk_2:
        alpha *= rho
        w = xk + alpha*dk
        F_w = F(w)
        mk += 1
        if mk >= 1000:
            raise Exception("line search trials exceeded")	  
    return alpha, w, F_w

def line_search_fw(F, xk, dk, beta=1, rho=0.78, sigma = 1e-3):
    sigma_norm_dk_2 = sigma*np.sum(dk*dk)
    alpha = beta
    w = xk + alpha*dk
    F_w = F(w)
    mk = 1
    while -np.sum(F_w*dk) < alpha*norm(F_w)*sigma_norm_dk_2:
        alpha *= rho
        w = xk + alpha*dk
        F_w = F(w)
        mk+=1
        if mk >= 1000:
            raise Exception("line search trials exceeded")
    return alpha, w, F_w

    
def zeta(Fw, xk, wk):
    '''Gives the quantity to project xk via xk - zeta*F(zk)
    onto the hyperplane {x in Rn| F(zk)^T(x - z_k) = 0}'''
    norm_Fw_squared = np.sum(Fw*Fw)
    zeta_k = np.sum(Fw*(xk - wk))/norm_Fw_squared
    return zeta_k


def cg_first_iteration(F, x0, P_c, F0, gamma):
    d0 = -F0
    alpha0, w0, Fw0 = line_search(F, x0, d0)
    x_1 = P_c(x0 - gamma*zeta(Fw0, x0, w0)*Fw0)
    return x_1, w0, Fw0

def cg_first_iteration_fw(F, x0, P_c, F0, gamma):
    d0 = -F0
    alpha0, w0, Fw0 = line_search_fw(F, x0, d0)
    x_1 = P_c(x0 - gamma*zeta(Fw0, x0, w0)*Fw0)
    return x_1, w0, Fw0


def compute_dk_mcg(Fk, dk_1, yk_1, mu = 0.01):
    ''' returns dk for the next iteration using the direction 
     from A.B. Abubakar et al Global convergence via descent
     modified three-term conjugate gradient projection algorithm
     with applications to signal recovery
     We assume k >= 1'''
    dk = -Fk
    denominator_choice1 = mu*norm(dk_1)*norm(yk_1)
    denominator_choice2 = np.sum(yk_1* dk_1)
    denominator = max(denominator_choice1, denominator_choice2)
    numerator = np.sum(Fk*yk_1)*dk_1 - np.sum(Fk*dk_1)*yk_1
    dk += numerator/denominator
    return dk

def compute_dk_TTCRM1(Fk, dk_1, yk_1, sk_1, m=0.7, sig=0.52, t=0.2):
    ''' returns dk for the next iteration using the direction from 
    Waziri et al - Adaptive three-term family of conjugate gradient
    residual methods for system of monotone nonlinear equations'''
    #sk_1 = 
    dk = -Fk
    ykb_1 = yk_1 - m*sk_1
    bk1 = np.sum(Fk* ykb_1)/np.sum(dk_1*ykb_1)
    bk1 += -sig*np.sum(ykb_1*ykb_1)*np.sum(Fk*sk_1)/(np.sum(dk_1*ykb_1)**2)
    bk1 += -t*np.sum(Fk*sk_1)/np.sum(dk_1*ykb_1)
    dk += bk1*dk_1
    dk += bk1*np.sum(Fk*dk_1)/np.sum(Fk*Fk)*Fk

    return dk

def TTCRM1(F, x0, Tol = 1e-8, P_c = lambda x: x):
    '''Implements Algorithm 3.1 Three-term Conjugate Risidual Method1
    from Waziri et al (2022) - Adaptive three-term family of conjugate 
    gradient residual method for system of monotone nonlinear equations'''
    k=0
    xk = x0
    max_iterations = 10000
    iflag = 0
    F = counter(F) #keeps track of function evaluations
    F0 = F(x0)
    norm_Fk = norm(F0)
    if norm_Fk <= Tol:
        return x0, k, norm_Fk, iflag
    dk = -F0
    xk, wk, Fwk = cg_first_iteration_fw(F, x0, P_c, F0, 1)  
    k+=1
    Fk = F(xk)
    norm_Fk = norm(Fk)
    sk_1 = wk - x0
    yk_1 = Fk - F0
    while norm_Fk > Tol:
        if k >= max_iterations:
       	    iflag = 1
            raise Exception('maximum iterations exceeded')
        dk = compute_dk_TTCRM1(Fk, dk, yk_1, sk_1)
        #assert dk is a descent direction
        assert inner(dk,Fk) < 0, f"dk not a descent direction"
        alpha_k, wk, Fwk = line_search_fw(F, xk, dk)
        zetak = zeta(Fwk, xk, wk)
        sk_1 = alpha_k * dk
        xk = xk - zetak*Fwk
        k+=1
        Fk_1 = Fk
        Fk = F(xk)
        norm_Fk = norm(Fk)
        yk_1 = Fk - Fk_1
        dk_1 = dk


    return xk, k, F.evals, norm_Fk

def mcg(F,x0, P_c, mu=0.01, gamma=1.8, Tol=1e-6):
    '''Implements algorithm 2.2 from the Abubakar et al (2019) paper 
    using the line search and beta_mcg methods'''
    k=0
    max_iterations = 10000
    iflag = 0
    F = counter(F) #keeps track of function evaluations
    F0 = F(x0)
    norm_Fk = norm(F0)
    if norm_Fk <= Tol:
        return x0, k, 0, norm_Fk
    xk, wk, Fwk = cg_first_iteration(F, x0, P_c, F0, gamma)
    k+=1
    dk_1 = -F0
    Fk = F(xk)
    norm_Fk = norm(Fk)
    yk_1 = Fwk - F0
    while norm_Fk > Tol:
        if k >= max_iterations:
            iflag = 1
            break
        #assert norm(yk_1) > Tol, f"wk and xk are the same"
        dk = compute_dk_mcg(Fk, dk_1, yk_1)
        #assert inner(dk, Fk) < 0 , f"dk not a descent direction"
        alphak, wk, Fwk = line_search(F, xk, dk)
        zetak = zeta(Fwk, xk, wk)
        xk = P_c(xk - gamma*zetak*Fwk)
        k+=1
        Fk_1 = Fk
        Fwk_1 = Fwk
        dk_1 = dk
        Fk = F(xk)
        norm_Fk = norm(Fk)
        yk_1 = Fwk_1 - Fk_1

    return xk, k, F.evals, norm_Fk 


