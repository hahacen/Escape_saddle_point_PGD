import numpy as np
from matplotlib import pyplot as plt
import math 
import time
from utility import ith_largest_singular_value
from utility import sample_from_ball

    
"""
define M* has shape (d,d), rank(M*) = r
U has shape (d,r)
randomly initialize U0

"""
d = 2
r = 2
# what's this c_max?
c_max = 0.5
max_itr = 1000

# multiply by 10 to avoid computation error by computer
U0 = np.random.uniform(low=-1, high=1, size=(d, r))
U0 = U0/(np.linalg.norm(U0, ord=2)*2)
# singular_values = np.random.rand(r)
singular_values = np.random.uniform(low=-0.5, high=0.5, size=r)
M_star = np.diag(singular_values)
M_star = np.pad(M_star, ((0, d - r), (0, d - r)), 'constant', constant_values=(0, 0))
Q, _ = np.linalg.qr(np.random.uniform(low=-1, high=1, size=(d, d)))
# M_star = Q @ M_star @ Q.T
M_star = [[4,0],[0,4]]

# Check the shape and rank of the matrices
# print(U0.shape, M_star.shape, np.linalg.matrix_rank(M_star))

sigma_1 = ith_largest_singular_value(M_star, 1)
sigma_r = ith_largest_singular_value(M_star, r)
Sigma_capital_sqrt = 2 * max(np.linalg.norm(U0, ord=2), 3 * (sigma_1 ** 0.5))
# Sigma_capital_sqrt = 0.3
# print("sigma:", Sigma_capital_sqrt)
# l-smooth
ell = 8*(Sigma_capital_sqrt**2)
# rho- Hessian Lipschitz
rho = 12*Sigma_capital_sqrt
# epsilon -SOSP
epsilon = (sigma_r ** 2) / (108 * Sigma_capital_sqrt)
# print(epsilon)
# print(ell)
# arbitrary c and delta
c = 0.3
delta = 0.1
Delta_f = r*Sigma_capital_sqrt ** 4 /2
# in case of PGDli algorithm
beta = 10*sigma_1





def obj_func(U, M_star = M_star):
    return 0.5 * np.linalg.norm(U @ U.T - M_star, 'fro')**2

def PGD(x0= U0, l = ell, _rho = rho, _eps = epsilon, _c = c, _delta = delta, _Delta_f = Delta_f, t_thres = 40):
    # traa = d*ell
    chi=3*max(np.log(d*l*_Delta_f/(_c*(_eps**2)*_delta)),4)
    eta=10*c/l
    radius=10000000*(_c**(1/2)*_eps)/(chi**2*l)
    g_thres=10000*c**(1/2)*epsilon/chi**2
    # f_thres=(c/(chi**3))*((epsilon**3/rho)**(1/2))
    # f_thres = c / chi**3 * (epsilon**3 / rho)**0.5
    f_thres = 1 / 2**3 * (0.1**3 / 2)**0.5
    # print(chi, eta, radius, g_thres, f_thres, epsilon)
    print("g_thres: ",g_thres)
    print("radius: ",radius)
    print("f_thres: ",f_thres)
    print("_Delta_f", _Delta_f)
    # print(rho)
    print("eta: ",eta)
    # if t_thres is None:
    #     # floor or ceiling?
    #     t_thres=math.floor(chi/c**2*l/(rho*epsilon)**(1/2))
    t_thres = 1
    t_noise= -t_thres-1 

    t =0
    xt = x0
    tilda_xt = None
    noise = 0
    while True:
        # if np.linalg.norm(gradient(xt),ord=2)<= g_thres and t-t_noise>t_thres:
        if t-t_noise>t_thres:
            tilda_xt = xt
            t_noise = t
            perturb  = sample_from_ball(d=d, radius=radius)
            # print("noise!")
            noise+=1
            xt = tilda_xt + perturb.reshape(d,1)
            print("xt:", xt)
        # print("t", t, " ", t_noise," ", t_thres)
        # print("tnoise", t_noise)
        # if t - t_noise == t_thres:
        #     print("f(xt): ", obj_func(xt))
        #     print("f(tilda_xt): ", obj_func(tilda_xt))
        # if t - t_noise == t_thres and obj_func(xt) - obj_func(tilda_xt) > -f_thres:
        #     return tilda_xt, t
        # if abs(obj_func(xt) - obj_func(tilda_xt)) < 0.01:
        #     print("exit: ", tilda_xt, t)
        #     return tilda_xt, t
        if t > max_itr:
            print(noise)
            return tilda_xt, t
        if abs(obj_func(xt) - obj_func(xt - eta*gradient(xt))) < 0.001:
            print("exit: ", xt, t)
            return xt, t
        xt = xt - eta*gradient(xt)
        t=t+1
        # print(noise)
        # print(xt, t)

def gradient(U, M_star = M_star):
    return U @ (U.T @ U) - M_star @ U

def evaluate(result_U, M_star=M_star):
    # how to evaluate?
    # what matrix?
    return obj_func(result_U, M_star = M_star)

result = PGD(x0= U0, l = ell, _rho = rho, _eps = epsilon, _c = c, _delta = delta, _Delta_f = Delta_f, t_thres = None)

# print(evaluate(result, M_star=M_star))
