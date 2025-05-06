import numpy as np
import numpy.linalg as LA
from itertools import count
from time import time
import scipy.linalg as scp_LA

"""
The following methods are based on the Chambolle and Pock PDA.
"""

# CP-PDA
def pd(J, prox_g, prox_f_conj, K,  x0, y0, sigma, tau, numb_iter=100):
    """
    Primal-dual algorithm of Pock and Chambolle for the problem min_x
    max_y [<Kx,y> + g(x) - f*(y)]
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    begin = time()  # time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0)]
    time_list = [time() - begin]
    for i in range(numb_iter):
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        z = x1 + theta * (x1 - x)
        y = prox_f_conj(y + sigma * K.dot(z), sigma)
        x = x1
        values.append(J(x, y))
        # time_list.append(time() - begin)
        time_list.append(time() - begin)
    end = time()  # time()
    
    
    print("----- Primal-dual method -----")
    print("Time execution:", end - begin)
    return [time_list, values, x, y]

# CP-PDA
def pd_new(J, prox_g, prox_f_conj, K,  x0, y0, sigma, tau, min_val, numb_iter=100, tol=1e-12):
    """
    Primal-dual algorithm of Pock and Chambolle for the problem min_x
    max_y [<Kx,y> + g(x) - f*(y)]
    J denotes some function which we compute in every iteration to
    study perfomance. It may be energy, primal-dual gap, etc.
    """
    begin = time()  # time()
    theta = 1.0
    x, y, z = x0, y0, x0
    values = [J(x0, y0, min_val)]
    time_list = [time() - begin]
    for i in range(numb_iter):
        x1 = prox_g(x - tau * K.T.dot(y), tau)
        z = x1 + theta * (x1 - x)
        y = prox_f_conj(y + sigma * K.dot(z), sigma)
        x = x1
        values.append(J(x, y, min_val))
        # time_list.append(time() - begin)
        time_list.append(time() - begin)
        err = values[-1]
        end = time()
        if err <= tol:
            print ("----- Primal-dual method -----")
            print ("Time execution:", round(end - begin,2))
            print ("Iteration:", i+1)
            print(err)  
            break
        
    if err > tol:
        print ("Golden-Ratio PDA does not terminate after", round(end - begin,2), "seconds")
        print(err)    
    end = time()  # time()

    return [values, x, y, time_list]

# CP-PDA-L
def pd_linesearch(J, prox_g, prox_f_conj, K,  x0, y0, tau, beta, min_val, numb_iter=100, tol = 1e-12):
    """
    Primal-dual method with linesearch for problem min_x max_y [
    <Kx,y> + g(x) - f*(y)].  Corresponds to Alg.1 in [].
    beta denotes sigma/tau from a classical primal-dual algorithm.
    """
    begin = time()
    theta = 1
    values = [J(x0, y0, min_val)]
    time_list = [time() - begin]
    mu = 0.7
    delta = 0.99
    iterates = [time_list, values, x0, y0, theta, tau, K.dot(x0)]
    sqrt_b = np.sqrt(beta)

    # function T is an operator that makes one iteration of the algorithm:
    # (x1, y1) = T(x,y, history)
    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y0 = prox_f_conj(y + tau * beta * Kz, tau * beta)
            if sqrt_b * tau * LA.norm(K.T.dot(y0 - y)) <= delta * LA.norm(y0 - y):
                break
            else:
                th *= mu
        values.append(J(x, y0, min_val))
        time_list.append(time() - begin)
        res = [time_list, values, x,  y0, th, tau, Kx]
        return res

    count_ = 0
    err = 1
    while count_ < numb_iter and err > tol:
        iterates = T(*iterates)
        err = iterates[1][-1]
        #print(err)
        count_ += 1
    # for i in range(numb_iter):
    #     iterates = T(*iterates)
    end = time()
    print("----- Primal-dual method with linesearch-----")
    print("Time execution:", round(end - begin,2))
    print("iteration:")
    print(len(iterates[1])-1)
    #print("opt:")
    #print(min(iterates[0]))
    print("error:")
    print(err)
    return iterates[:4]

# CP-PDA-L
def pd_linesearch_dual_is_square_norm(J, prox_g, b, K,  x0, y1, tau, beta, min_val, numb_iter=100, tol = 1e-12):
    """
    Primal-dual method with linesearch for min_x max_y [ <Kx,y> + g(x) - f*(y) ]
    for the case when f*(y) = 0.5||y-b||^2
    """
    
    theta = 1
    values = [J(x0, y1, min_val)]
    time_list = [0]
    mu = 0.7
    delta = 0.99
    Kx0 = K.dot(x0)
    iterates = [time_list, values, x0, y1,
                theta, tau, Kx0, K.T.dot(Kx0), K.T.dot(y1)]
    sqrt_beta = np.sqrt(beta)
    KTb = K.T.dot(b)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old, KTKx_old, KTy):
        # Note that KTy = K.T.dot(y)
        x = prox_g(x_old - tau_old * KTy, tau_old)
        Kx = K.dot(x)
        KTKx = K.T.dot(Kx)
        th = np.sqrt(1 + th_old)
        for j in count(0):
            tau = tau_old * th
            sigma = tau * beta
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = (y + sigma * (Kz + b)) / (1. + sigma)
            KTKz = (1 + th) * KTKx - th * KTKx_old
            KTy1 = (KTy + sigma * (KTKz + KTb)) / (1. + sigma)
            if sqrt_beta * tau * LA.norm(KTy1 - KTy) <= delta * LA.norm(y1 - y):
                break
            else:
                th *= mu
        values.append(J(x, y1, min_val))
        time_list.append(time() - begin)
        res = [time_list, values, x, y1, th, tau, Kx, KTKx,  KTy1]
        return res

    count_ = 0
    err = 1
    begin = time()
    while count_ < numb_iter and err > tol:
        iterates = T(*iterates)
        err = iterates[1][-1]
        #print(err)
        count_ += 1

    end = time()
    print(
        "----- Primal-dual method with  linesearch. f^*(y)=0.5*||y-b||^2-----")
    print("Time execution:", end - begin)
    print("iteration:")
    print(len(iterates[1])-1)
    #print("opt:")
    #print(min(iterates[0]))
    print("error:")
    print(err)
    # return [iterates[i] for i in [0, -1]]
    return iterates[:4]

# CP-PDA-L
def pd_linesearch_general(J, prox_g, prox_f_conj, h, dh, K,  x0, y1, tau, beta, numb_iter=100):
    """
    Primal-dual method with  linesearch for problem min_x max_y [
    <Kx,y> + g(x) - f*(y)-h(y)].  Corresponds to Alg.4 in the paper.
    """
    begin = time()
    theta = 1
    values = [J(x0, y1)]
    time_list = [time() - begin]
    mu = 0.7
    delta = 0.99
    iterates = [time_list, values, x0, y1, theta, tau, K.dot(x0)]
    sqrt_b = np.sqrt(beta)

    # function T is an operator that makes one iteration of the algorithm:
    # (x1, y1) = T(x,y, history)

    def T(time_list, values, x_old, y, th_old, tau_old, Kx_old):
        x = prox_g(x_old - tau_old * K.T.dot(y), tau_old)
        Kx = K.dot(x)
        th = np.sqrt(1 + th_old)
        dhy = dh(y)
        hy = h(y)
        for j in count(0):
            tau = tau_old * th
            sigma = beta * tau
            z = x + th * (x - x_old)
            Kz = Kx + th * (Kx - Kx_old)
            y1 = prox_f_conj(y + sigma * (Kz - dhy), sigma)
            hy1 = h(y1)
            if sigma * tau * (LA.norm(K.T.dot(y1 - y)))**2 + 2 * sigma * (hy1 - hy - dhy.dot(y1 - y)) <= delta * np.dot(y1 - y, y1 - y):
                break
            else:
                th *= mu
        # print(j, tau)
        values.append(J(x, y1))
        time_list.append(time() - begin)
        res = [time_list, values, x,  y1, th, tau, Kx]
        return res

    for i in range(numb_iter):
        iterates = T(*iterates)

    end = time()
    print("----- General primal-dual method with linesearch-----")
    print("Time execution:", end - begin)
    return iterates[:4]