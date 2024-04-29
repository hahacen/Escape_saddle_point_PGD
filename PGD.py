import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
import math 
import time as ttt
from utility import ith_largest_singular_value
from utility import sample_from_ball

    
"""
define M* has shape (d,d), rank(M*) = r
U has shape (d,r)
randomly initialize U0

"""
d = 2
r = 1
# what's this c_max?
c_max = 0.5
max_itr = 100000


def single_run(d=d, r=r, t_thresin= 1):
    U0 = np.random.uniform(low=-1, high=1, size=(d, r))
    U0 = U0/(np.linalg.norm(U0, ord=2)*2)
    # singular_values = np.random.rand(r)
    singular_values = np.random.uniform(low=-0.5, high=0.5, size=r)
    M_star = np.diag(singular_values)
    M_star = np.pad(M_star, ((0, d - r), (0, d - r)), 'constant', constant_values=(0, 0))
    Q, _ = np.linalg.qr(np.random.uniform(low=-1, high=1, size=(d, d)))
    M_star = Q @ M_star @ Q.T
    # M_star = [[4,0],[0,4]]

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
    # arbitrary c and delta
    c = 0.3
    delta = 0.1
    Delta_f = r*Sigma_capital_sqrt ** 4 /2
    # in case of PGDli algorithm
    beta = 10*sigma_1


    result, t = PGD( M_star = M_star,x0= U0, l = ell, _rho = rho, _eps = epsilon, _c = c, _delta = delta, _Delta_f = Delta_f, t_thres = t_thresin)
    return result, t
    print(result)
    print(M_star)

def integrate_run():
    # result, time = single_run(t_thresin= 2)
    # result, time = single_run(t_thresin= 4)
    average_iterations = []
    success_rate = []
    time_ex = []
    # t_thres = [1, 10, 40, 100, 300, 500, 1000]
    t_thres = [1, 5, 10, 40, 100, 300, 500, 1000, 5000, 10000, 20000, 30000]
    # t_thres = [1, 40, 5000, 10000, 30000]
    for t in t_thres:
        count = 0
        escape_rate = 0
        av_it = 0
        print("tthres:", t)
        excute_time = 0
        while True:
            start_time = ttt.time()
            result, time = single_run(t_thresin= t)
            count = count+1
            # if successfully escpae the saddle point
            if time < max_itr-1:
                escape_rate = escape_rate+1
                av_it = av_it+time
                elapsed_time = ttt.time() - start_time  # Calculate elapsed time
                excute_time += elapsed_time  # Accumulate the total execution time
            if count >=40:
                if escape_rate ==0:
                    average_iterations.append(max_itr)
                else:
                    average_iterations.append(av_it/escape_rate)
                success_rate.append(escape_rate/count)
                time_ex.append(excute_time/escape_rate)
                break
        print(average_iterations)
        print(success_rate)
        print(time_ex)
    # Creating the figure and the axes for the subplots
    def forward(x):
        return x**(1/5)

    def inverse(x):
        return x**5
    
    plt.figure(figsize=(8, 5))
    plt.plot(t_thres, average_iterations, marker='o')
    plt.title('Average Iterations vs. Threshold')
    plt.xlabel('Threshold (t_thres)')
    plt.ylabel('Average Iterations')
    plt.grid(True)
    plt.xscale('function', functions=(forward, inverse))
    plt.show()

    # Creating the second figure for success rate
    plt.figure(figsize=(8, 5))
    plt.plot(t_thres, success_rate, marker='o', color='green')
    plt.title('Success Rate vs. Threshold')
    plt.xlabel('Threshold (t_thres)')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.xscale('function', functions=(forward, inverse))
    plt.show()

def obj_func(U, M_star ):
    return 0.5 * np.linalg.norm(U @ U.T - M_star, 'fro')**2

def PGD(M_star, x0, l , _rho , _eps , _c, _delta , _Delta_f , t_thres = 40):
    # traa = d*ell
    chi=3*max(np.log(d*l*_Delta_f/(_c*(_eps**2)*_delta)),4)
    eta=_c/l
    radius=(_c**(1/2)*_eps)/(chi**2*l)
    g_thres=_c**(1/2)*_eps/chi**2
    # f_thres=(c/(chi**3))*((epsilon**3/rho)**(1/2))
    # f_thres = c / chi**3 * (epsilon**3 / rho)**0.5
    f_thres = 1 / 2**3 * (0.1**3 / 2)**0.5
    # print(chi, eta, radius, g_thres, f_thres, epsilon)
    # print("g_thres: ",g_thres)
    # print("radius: ",radius)
    # print("f_thres: ",f_thres)
    # print("_Delta_f", _Delta_f)
    # # print(rho)
    # print("eta: ",eta)
    if t_thres is None:
        # floor or ceiling?
        t_thres=math.floor(chi/_c**2*l/(_rho*_eps)**(1/2))
    # t_thres = 40
    t_noise= -t_thres-1 

    t =0
    xt = x0
    tilda_xt = None
    noise = 0
    while True:
        if np.linalg.norm(gradient(xt, M_star),ord=2)<= g_thres and t-t_noise>t_thres:
        # if t-t_noise>t_thres:
            tilda_xt = xt
            t_noise = t
            perturb  = sample_from_ball(d=d, radius=radius)
            # print("noise!")
            # print(perturb)
            noise+=1
            xt = tilda_xt + perturb.reshape(d,1)
            # print("xt:", xt)
        if t - t_noise == t_thres and obj_func(xt, M_star) - obj_func(tilda_xt, M_star) > -f_thres:
            return tilda_xt, t
        # if t > max_itr:
        #     print(noise)
        #     return tilda_xt, t
        # if obj_func(xt) - obj_func(xt - eta*gradient(xt)) < 0.001:
        #     print(xt)
        #     return xt
        xt = xt - eta*gradient(xt, M_star)
        t=t+1
        if t>max_itr:
            return xt, max_itr
        # print(noise)
        # print("tthres:", t_thres)
        # print(xt, t)

def gradient(U, M_star ):
    return U @ (U.T @ U) - M_star @ U

# result = PGD(x0= U0, l = ell, _rho = rho, _eps = epsilon, _c = c, _delta = delta, _Delta_f = Delta_f, t_thres = 1)
# single_run()
# print(evaluate(result, M_star=M_star))
# print("M_star: ", M_star)

integrate_run()