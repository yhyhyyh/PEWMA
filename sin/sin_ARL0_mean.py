import multiprocessing
import numpy as np
import math
import random

np.random.seed(1)

def calculate_Boundary(A,omega,theta0, m0, p):

    c1 = 13.8065
    c2 = 11.8532
    c3 = 26.4037

    q = np.array(np.zeros((p - 1, 1)))
    theta_list = []
    Y_list = []
    for i in range(m0):
        nt = 2 * c1 / (1 + np.exp(-(i - c2) / c3))  # 人口规模增加
        # nt = ((c1 / 2.4)/(1 + np.exp((i - c2) / c3))) +18  # 人口规模减少
        # nt = np.random.uniform(15,20)  # 人口规模均匀分布
        theta1=A*(math.sin(omega*i))+theta0
        # print(theta[i])
        x0 = np.random.poisson(nt * theta1)  # 真实的分布是泊松分布
        theta = x0 / nt
        # print(theta)
        theta_list.append(theta)
        for j in range(p - 1):
            q[j] = np.percentile(theta_list, ((j + 1) / p) * 100)  # 计算区间分界点

    for i in theta_list:
        Y = np.array(np.zeros((p, 1)))
        if i < q[0]:
            Y[0] = 1
        elif i > q[p - 2]:
            Y[p - 1] = 1
        for k in range(2, p):
            if q[k - 2] <= i < q[k - 1]:
                Y[k - 1] = 1
        Y_list.append(Y)
    f0=np.sum(Y_list, axis=0)/m0
    return q,f0
# print(calculate_Boundary(A=0.1,omega=0.5,theta0=1, m0=500, p=5))
q,f0=calculate_Boundary(A=0.1,omega=0.5,theta0=1, m0=500, p=5)

def calculate_ARL(rep,T,tau,Lambda,p,hp):
    global t
    RL = np.mat(np.zeros((rep, 1)))
    ind = 0
    S_RL = []
    for v in range(rep):
        G = np.mat(np.zeros((p, T + 1)))
        E = np.mat(np.zeros((p, T + 1)))
        U = np.zeros((1, T + 1))
        S_obs = np.mat(np.zeros((p, T + 1)))
        S_exp = np.mat(np.zeros((p, T + 1)))
        for t in range(1, T + 1):
            E[:, 0] = f0
            S_obs[:, 0] = S_exp[:, 0] = 0
            Y = np.random.multinomial(1, f0.flatten())
            Y = Y.reshape((p, 1))
            Y = Y + np.random.normal(0, 0.01, size=Y.shape)
            G[:, t] = Y
            E[:, t] = ((1 - Lambda) * E[:, t - 1]) + (Lambda * G[:, t])
            S_obs[:, t] = S_obs[:, t - 1] + E[:, t]
            S_exp[:, t] = S_exp[:, t - 1] + f0
            U[:, t] = ((S_obs[:, t] - S_exp[:, t]).T) * (np.mat(np.linalg.pinv(np.diagflat(S_exp[:, t])))) * (S_obs[:, t] - S_exp[:, t])
            # print('U[:, t]=', U[:, t])
            if U[:, t] > hp:
                break
        # print('-------------------i------------------', t)
        if t > tau and t <= T:
            RL = t - tau
            ind = ind + 1
        else:
            RL = 0
        S_RL.append(RL)
    for i in range(len(S_RL) - 1, -1, -1):  # 删除0值
        if S_RL[i] == 0:
            S_RL = np.delete(S_RL, i)
    # print(S_RL)
    if len(S_RL) > 0:
        ARL = np.mean(S_RL)
        SDRL = np.std(S_RL)
        SE = SDRL / math.sqrt(len(S_RL))
    else:
        ARL = 0
        SDRL = 0
        SE = 0
    print('ARL=', ARL)
    print('SE=', SE)
    print('SDRL=', SDRL)
    print('ind=', ind)
    return ARL
