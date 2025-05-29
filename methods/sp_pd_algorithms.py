from __future__ import division
import numpy as np
import numpy.linalg as LA
from itertools import count
from time import time
import scipy.linalg as scp_LA
from time import perf_counter

def adaptive_graals_acc(J, F, prox_g, x1, numb_iter=100, phi=1.5, output=False):
    """ Adaptive Golden Ratio algorithm.
    Input 
    -----
    J : function that computes residual in every iteration.
        Takes x as input.
    F : main operator.
        Takes x as input.
    prox_g: proximal operator.
        Takes two parameters x and a scalar as input.
    x1: Starting point.
        np.array, must be consistent with J, F and prox_g.
    numb_iter: number of iteration to run rhe algorithm.
    phi: a key parameter for the algorithm.
          Must be between 1 and the golden ratio, 1.618... Choice
          phi=1.5 seems to be one of the best.
    output: boolean.  
          If true, prints the length of a stepsize in every iteration.  
          Useful for monitoring.

    Return
    ------
    values: 1d array
          Collects all values that were computed in every iteration
          with J(x)
    x, x_ : last iterates
    time_list: list of time stamps in every iteration. 
          Useful for monitoring.
    """
    begin = perf_counter()
    rho = 1.1
    xi = 1.5
    gamma = 1
    la = 1
    beta = 0.9
    theta = 1
    
    x, x_ = x1.copy(), x1.copy()
    x0 = x + np.random.randn(x.shape[0]) * 1e-9
    Fx = F(x)
    la = phi / 2 * LA.norm(x - x0) / LA.norm(Fx - F(x0))
    rho = 1. / phi + 1. / phi**2
    values = [J(x)]
    time_list = [perf_counter() - begin]
    th = 1

    for i in range(numb_iter):
        phi_ = (phi-rho)/(phi+rho*gamma*la)
        beta = (1+gamma*phi_*la)*beta
        omega = 2 - xi/phi - phi**2 * rho / (1+phi)
        alpha = omega * theta + gamma * la
        x1 = prox_g(x_ - la * Fx, la)
        Fx1 = F(x1)
        psi = (phi - rho) / (phi+rho*gamma*la)
        n1 = LA.norm(x1 - x)**2
        n2 = LA.norm(Fx1 - Fx)**2
        n1_div_n2 = n1/n2 if n2 != 0 else la*10

        la1 = min(rho * la, 0.25 * phi * th / la * n1_div_n2)
        x_ = ((phi - 1) * x1 + x_) / phi
        if output:
            print (i, la)
        th = phi * la1 / la
        x, la, Fx = x1, la1, Fx1
        values.append(J(x))
        time_list.append(perf_counter() - begin)
    end = perf_counter()

    print("CPU time for aGRAAL:", end - begin)
    return values, x, x_, time_list

# SP-PDA
def pd_sp(J, prox_g, prox_f_conj, K, x0, y0, sigma, tau, min_val, numb_iter=100, tol=1e-12):
    """
    The Step free Primal-dual algorithm for the problem min_x max_y [<Kx,y> + g(x) - f*(y)] 
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    w0 = K.dot(x0)
    m,n = K.shape
    Kty_old = K.T.dot(y0)
 
    I1 = np.eye(m)
    I1.shape
    M = (sigma*tau) * np.dot(K, K.T) + (1+0.001) * I1
    L = LA.cholesky(M)
    
    values = [J(x0, y0, min_val)]
    tt = [0]  

    iterates = [values, x0, w0, y0] +[Kty_old] + [tt]
    def T(values, x_old, w_old, y_old, Kty_old, tt):
        # compute x
        Kx0 = K.dot(x_old)
        x = prox_g(x_old - tau * Kty_old, tau)
        Kx = K.dot(x)
            
        # compute w
        y1 = prox_f_conj(sigma*w_old + y_old, sigma)
        w = w_old + 1/sigma * (y_old  - y1)

        # compute y
        h_hat = 2*Kx-Kx0-(2*w-w_old)
        # print(np.isinf(L).any(),np.isinf(y_hat).any)
        z = scp_LA.solve_triangular(L, h_hat, lower=True)
        # print(np.isinf(y1).any())
        h = scp_LA.solve_triangular(L.T, z, lower=False)
        y = y_old + sigma * h
        Kty = K.T.dot(y)

        values.append(J(x, y1, min_val))
        #values.append(J(x, Kx))
        tt.append(time() - begin)
        res = [values, x, w, y, Kty,tt]
        return res
        
    count_ = 0
    begin = time()
    err = 1
    while count_ < numb_iter and err > tol:
        iterates = T(*iterates)
        err = iterates[0][-1]
        #print(err)
        count_ += 1
            
    end = time()
    print ("----- SP PDA-----")
    print ("Time execution:", round(end - begin,2))
    print("iteration:")
    print(len(iterates[-1])-1)
    #print("opt:")
    #print(min(iterates[0]))
    print("error:")
    print(err)
    return [iterates[i] for i in [0, -1]]

def pd_sp_msmall(J, prox_g, prox_f_conj, K, x0, y0, sigma, tau, min_val, numb_iter=100, tol=1e-12):
    """
    The Step free Primal-dual algorithm for the problem min_x max_y [<Kx,y> + g(x) - f*(y)] 
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    w0 = K.dot(x0)
    m,n = K.shape
    Kty_old = K.T.dot(y0)
 
    I1 = np.eye(m)
    I1.shape
    M = (sigma*tau) * np.dot(K, K.T) + (1+0.001) * I1
    # L = LA.cholesky(M)
    M_inver = np.linalg.inv(M)
    values = [J(x0, y0, min_val)]
    tt = [0]  

    iterates = [values, x0, w0, y0] +[Kty_old] + [tt]
    def T(values, x_old, w_old, y_old, Kty_old, tt):
        # compute x
        x = prox_g(x_old - tau * Kty_old, tau)
        Kx = K.dot(x)
        Kx0 = K.dot(x_old)    
        # compute w
        # w = prox_f_conj(w_old + y_old/sigma, sigma)
        y1 = prox_f_conj(sigma*w_old + y_old, sigma)
        w = w_old + 1/sigma * (y_old  - y1)

        # compute y
        h_hat = sigma * (2*Kx-Kx0-(2*w-w_old))
        # print(np.isinf(L).any(),np.isinf(y_hat).any)
        # z = scp_LA.solve_triangular(L, h_hat, lower=True)
        # print(np.isinf(y1).any())
        # h = scp_LA.solve_triangular(L.T, z, lower=False)
        y = y_old + np.dot(M_inver, h_hat)
        Kty = K.T.dot(y)

        values.append(J(x, y1, min_val))
        #values.append(J(x, Kx))
        tt.append(time() - begin)
        res = [values, x, w, y, Kty,tt]
        return res
        
    count_ = 0
    begin = time()
    err = 1
    while count_ < numb_iter and err > tol:
        iterates = T(*iterates)
        err = iterates[0][-1]
        #print(err)
        count_ += 1
            
    end = time()
    print ("----- SP PDA-----")
    print ("Time execution:", round(end - begin,2))
    print("iteration:")
    print(len(iterates[-1])-1)
    #print("opt:")
    #print(min(iterates[0]))
    print("error:")
    print(err)
    return [iterates[i] for i in [0, -1]]

def pd_sp_all(J, prox_g, prox_f_conj, dis_sub,K, x0, y0, sigma, tau, min_val, la,b,numb_iter=100, tol=1e-12):
    """
    The Balanced-Golden-Ratio Primal-dual algorithm for the problem min_x max_y [<Kx,y> + g(x) - f*(y)] 
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    w0 = K.dot(x0)
    m,n = K.shape
 
    I1 = np.eye(m)
    M = (sigma*tau) * np.dot(K, K.T) + (1+0.001) * I1
    
    values = [J(x0, y0, min_val)]
    tt = [0]  
    Kty_old = K.T.dot(y0)

    iterates = [values, x0, w0, y0] +[Kty_old] + [tt]
    L = LA.cholesky(M)
    def T(values, x_old, w_old, y_old, Kty_old, tt):
        # compute x
        Kx0 = K.dot(x_old)
        x = prox_g(x_old - tau * Kty_old, tau)
        Kx = K.dot(x)
            
        # compute w
        y1 = prox_f_conj(sigma*w_old + y_old, sigma)
        w = w_old + 1/sigma * (y_old  - y1)
    
        # compute y
        h_hat = 2*Kx-Kx0-(2*w-w_old)
        # print(np.isinf(L).any(),np.isinf(y_hat).any)
        z = scp_LA.solve_triangular(L, h_hat, lower=True)
        # print(np.isinf(y1).any())
        h = scp_LA.solve_triangular(L.T, z, lower=False)
        y = y_old + sigma * h
        Kty = K.T.dot(y)
              
        p_inf = LA.norm(Kx - w,1)
        d_inf = dis_sub(la, x, Kty) + LA.norm(w - y -b,1)

        values.append(max(p_inf,d_inf))
        tt.append(time() - begin)
        res = [values, x, w, y, Kty, tt]
        return res
        
    count_ = 0
    begin = time()
    err = 1
    while count_ < numb_iter and err > tol:
        iterates = T(*iterates)
        err = iterates[0][-1]
        #print(err)
        count_ += 1
            
    end = time()
    print ("----- SP PDA-----")
    print ("Time execution:", round(end - begin,2))
    print("iteration:")
    print(len(iterates[-1])-1)
    # print("opt:")
    # print(min(iterates[0]))
    print("error:")
    print(err)
    return [iterates[i] for i in [0,1, -1]]
    
