from math import sqrt
import numpy as np
from numpy.random import uniform
import math
import random


# Written by KJ
def grad_projected(arrival_rates, service_rates, delta, initial_a):
    prsc = 'float64'
    N = 50000
    ep = 0.00000000001
    gamma = 0.01
    # parameter end

    n1 = len(arrival_rates)
    n2 = len(service_rates)
    mu = np.array(service_rates, dtype=prsc)
    lbd = np.array(arrival_rates, dtype=prsc)
    a = np.array(initial_a, dtype=prsc)

    delta = np.array(delta, dtype=prsc)

    for x in range(N):
        I = []
        for i in range(n1):
            for j in range(n2):
                if a[i][j] < ep:
                    I.append(n2 * i + j)
        ps = 1
        ter = 0
        while ps == 1:
            NMT = np.array([[0] * (n1 * n2)] * (n1 + len(I)), dtype=prsc)
            for i in range(n1):
                for j in range(n2):
                    NMT[i][n2 * i + j] = 1
            cnt = 0
            for k in I:
                NMT[n1 + cnt][k] = 1
                cnt = cnt + 1
            NM = np.transpose(NMT)
            tmpM = np.linalg.inv(np.matmul(NMT, NM))

            PM = np.identity(n1 * n2) - np.matmul(NM, np.matmul(tmpM, NMT))
            Del = np.array([0] * (n1 * n2), dtype=prsc)
            for j in range(n2):
                lbd2 = 0
                for i in range(n1):
                    lbd2 = lbd2 + lbd[i] * a[i][j]
                for i in range(n1):
                    Del[n2 * i + j] = lbd[i] * (mu[j] / ((mu[j] - lbd2) ** 2) + delta[i][j])
            s = -np.matmul(PM, Del)
            lbd3 = np.matmul(np.matmul(tmpM, NMT), Del)
            ps = 0
            if np.linalg.norm(s) < ep:
                subI = I.copy()
                for i in range(len(I)):
                    if lbd3[n1 + i] < 0:
                        subI.remove(I[i])
                        ps = 1
                I = subI.copy()
                if ps == 0:
                    ter = 1
        if ter == 1:
            break
        if np.linalg.norm(s) > 1:
            s = s / np.linalg.norm(s)
        gamma2 = gamma * 2 * random.random()
        for i in range(n1):
            for j in range(n2):
                if n2 * i + j in I:
                    continue
                if s[n2 * i + j] >= 0:
                    continue
                else:
                    if gamma2 >= -a[i][j] / s[n2 * i + j]:
                        gamma2 = -a[i][j] / s[n2 * i + j]
        for j in range(n2):
            lbd2 = 0
            disp = 0
            for i in range(n1):
                lbd2 = lbd[i] * a[i][j]
                disp = s[n2 * i + j] * lbd[i]
            if disp <= 0:
                continue
            else:
                if gamma2 >= (mu[j] - lbd2) / disp:
                    gamma2 = (mu[j] - lbd2) / disp
        a = a + gamma2 * np.reshape(s, (n1, n2))
        for i in range(n1):
            jmax = 0
            for j in range(n2):
                if a[i][j]>a[i][jmax]:
                    jmax = j
            for j in range(n2):
                if a[i][j]<0:
                    a[i][jmax] = a[i][jmax]+a[i][j]
                    a[i][j] = 0
        gamma = gamma*0.9999
    F = 0
    for j in range(n2):
        lbd2 = 0
        for i in range(n1):
            lbd2 = lbd2 + lbd[i] * a[i][j]
        F = F + lbd2 / (mu[j] - lbd2)

    for i in range(n1):
        for j in range(n2):
            F = F + lbd[i] * delta[i][j] * a[i][j]
    F = F/sum(arrival_rates)

    result = {'A': a, 'Mean_completion_time': F}
    return result