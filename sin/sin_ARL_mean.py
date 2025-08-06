import numpy as np
import math
import random
from concurrent.futures import ProcessPoolExecutor

np.random.seed(1)

def calculate_Boundary(A, omega, theta0, m0, p):
    c1 = 13.8065
    c2 = 11.8532
    c3 = 26.4037

    q = np.array(np.zeros((p - 1, 1)))
    theta_list = []
    Y_list = []

    for i in range(m0):
        # nt = 2 * c1 / (1 + np.exp(-(i - c2) / c3))  # 人口规模增加
        # nt = ((c1 / 2.4)/(1 + np.exp((i - c2) / c3))) +18  # 人口规模减少
        nt = np.random.uniform(15,20)  # 人口规模均匀分布
        theta1 = A * (math.sin(omega * i)) + theta0
        x0 = np.random.poisson(nt * theta1)
        # x0 = x0 + np.random.normal(0, 0.01)
        theta = x0 / nt
        theta_list.append(theta)
        for j in range(p - 1):
            q[j] = np.percentile(theta_list, ((j + 1) / p) * 100)

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
    f0 = np.sum(Y_list, axis=0) / m0
    return q, f0
q,f0=calculate_Boundary(A=0.1, omega=0.5, theta0=1, m0=500, p=2)

def calculate_ARL(A, omega, theta0, rep, T, tau, shift, Lambda, p, hp):
    global t
    RL = np.mat(np.zeros((rep, 1)))
    ind = 0
    S_RL = []
    for v in range(rep):
        c1 = 13.8065
        c2 = 11.8532
        c3 = 26.4037

        nt = np.zeros((1, T + 1))
        x = np.zeros((1, T + 1))

        G = np.mat(np.zeros((p, T + 1)))
        E = np.mat(np.zeros((p, T + 1)))
        U = np.mat(np.zeros((1, T + 1)))
        S_obs = np.mat(np.zeros((p, T + 1)))
        S_exp = np.mat(np.zeros((p, T + 1)))

        for t in range(1, T + 1):
            Y = np.mat(np.zeros((p, 1)))
            E[:, 0] = f0
            S_obs[:, 0] = S_exp[:, 0] = 0

            # nt[:, t] = 2 * c1 / (1 + np.exp(-(t - c2) / c3))  # 人口规模增加
            # nt[:, t] = ((c1 / 2.4)/(1 + np.exp((t - c2) / c3))) +18  # 人口规模减少
            nt[:, t] = np.random.uniform(15,20)  # 人口规模均匀分布

            if t <= tau:
                x[:, t] = np.random.poisson(nt[:, t] * (A * (math.sin(omega * t)) + theta0))
            else:
                x[:, t] = np.random.poisson(nt[:, t] * (A * (math.sin(omega * t)) + (theta0+shift)))
            # x[:, t] = x[:, t] + np.random.normal(0, 0.01)
            theta = x[:, t] / nt[:, t]

            if theta < q[0]:
                Y[0] = 1
            elif theta > q[p - 2]:
                Y[p - 1] = 1
            for k in range(2, p):
                if q[k - 2] <= theta < q[k - 1]:
                    Y[k - 1] = 1
            Y = Y + np.random.normal(0, 0.01, size=Y.shape)

            G[:, t] = Y
            E[:, t] = ((1 - Lambda) * E[:, t - 1]) + (Lambda * G[:, t])
            S_obs[:, t] = S_obs[:, t - 1] + E[:, t]
            S_exp[:, t] = S_exp[:, t - 1] + f0
            U[:, t] = ((S_obs[:, t] - S_exp[:, t]).T) * (np.mat(np.linalg.pinv(np.diagflat(S_exp[:, t])))) * (S_obs[:, t] - S_exp[:, t])
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
        ARL = sum(S_RL) / ind
        SDRL = np.std(S_RL,ddof=1)
        SE = SDRL / math.sqrt(len(S_RL))
        # Q10 = np.percentile(S_RL, 10)
        # Med = np.median(S_RL)
        # Q90 = np.percentile(S_RL, 90)
        # FAR = sum(S_RL <= 50, 1) / len(S_RL)
    else:
        ARL = 0
        SDRL = 0
        SE = 0
        # Q10 = 0
        # Med = 0
        # Q90 = 0
        # FAR = 0
    print('ARL=', ARL)
    print('SE=', SE)
    print('SDRL=', SDRL)
    # print('Q10=', Q10)
    # print('Med=', Med)
    # print('Q90=', Q90)
    # print('ind=', ind)
    # print('FAR=', FAR)
    return ARL
print(calculate_ARL(A=0.1, omega=0.5, theta0=1, rep=2000, T=1000, tau=50, shift=0.01, Lambda=0.1, p=2, hp=0.687255859375))

# 5.4453125
# 5.33013916015625
# 4.586174011230469
# 4.703125 ARL0

# 5.982933044433594500

# 4.864006042480469 p=5 lambda=0.05
# 5.6250762939453125 p=5 lambda=0.15
# 5.6252288818359375 p=5 lambda=0.2
# 5.5892181396484375 p=5 lambda=0.25
# 5.527343153953552 p=5 lambda=0.3

# 0.687255859375 p=2
# 2.6384735107421875 p=4