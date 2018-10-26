# Written by JS
def barrier_method(arrival_rates, service_rates, delta, initial_a, eps1 = 1e-7, eps2 = 1e-7, t0 = 1, m = 1.1, alpha = 0.01, beta = 0.1):
    arrival_rates = np.array(arrival_rates, dtype=np.float64)
    service_rates = np.array(service_rates)
    delta = np.array(delta)

    Ml = len(arrival_rates)
    Nl = len(service_rates)
    t = t0

    # F is an Nl by Nl matrix which represents the null-space of [1 1 ... 1]
    F = np.zeros((Nl, Nl))
    for i in range(Nl):
        F[i, i] = 1
    for i in range(Nl - 1):
        F[i + 1, i] = -1
    F[0, Nl - 1] = -1

    # Initial guess
    initial_a = np.array(initial_a)
    Y = np.zeros((Ml, Nl))
    for i in range(Ml):
        Y[i, :] = np.linalg.lstsq(F, initial_a[i, :] - 1 / Nl * np.ones(Nl), rcond=None)[0]
        # Define f, gradf, hessf

    # f
    def f(k, Z):
        S11 = 0
        S12 = 0
        S21 = 0
        S22 = 0
        S31 = 0
        S32 = 0
        S41 = 0
        S42 = 0
        S51 = 0
        S52 = 0
        # 1st sum
        for j in range(Nl - 1):
            for i in range(Ml):
                S12 += (Z[i, j + 1] - Z[i, j] + 1 / Nl) * arrival_rates[i]
            S11 += -1 + service_rates[j + 1] / (service_rates[j + 1] - S12)
            S12 *= 0
        for i in range(Ml):
            S12 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
        S11 += -1 + service_rates[0] / (service_rates[0] - S12)
        # 2nd sum
        for j in range(Nl - 1):
            for i in range(Ml):
                S22 += arrival_rates[i] * delta[i, j + 1] * (Z[i, j + 1] - Z[i, j] + 1 / Nl)
            S21 += S22
            S22 *= 0
        for i in range(Ml):
            S22 += arrival_rates[i] * delta[i, 0] * (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl)
        S21 += S22
        # 3rd sum
        for j in range(Nl - 1):
            for i in range(Ml):
                S32 += math.log(Z[i, j + 1] - Z[i, j] + 1 / Nl)
            S31 += S32
            S32 *= 0
        for i in range(Ml):
            S32 += math.log(Z[i, 0] - Z[i, Nl - 1] + 1 / Nl)
        S31 += S32
        # 4th sum
        for j in range(Nl - 1):
            for i in range(Ml):
                S42 += math.log(1 - (Z[i, j + 1] - Z[i, j] + 1 / Nl))
            S41 += S42
            S42 *= 0
        for i in range(Ml):
            S42 += math.log(1 - (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl))
        S41 += S42
        # 5th sum
        for j in range(Nl - 1):
            for i in range(Ml):
                S52 += Z[i, j + 1] - Z[i, j] + 1 / Nl
            S51 += math.log(service_rates[j + 1] - S52)
            S52 *= 0
        for i in range(Ml):
            S52 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
        S51 += math.log(service_rates[0] - S52)
        # total
        return k * (S11 + S21) - S31 - S41 - S51

    # gradf
    def gradf(k, Z):
        grad = np.zeros(Ml * Nl)
        T1 = 0
        T2 = 0
        T3 = 0
        T4 = 0
        # Z = np.c_[Z, Z[:, [0,1]]]
        for be in range(Nl - 2):
            for al in range(Ml):
                for i in range(Ml):
                    T1 += ((Z[i, be + 1] - Z[i, be] + 1 / Nl) * arrival_rates[i])
                for i in range(Ml):
                    T2 += ((Z[i, be + 2] - Z[i, be + 1] + 1 / Nl) * arrival_rates[i])
                for i in range(Ml):
                    T3 += Z[i, be + 1] - Z[i, be] + 1 / Nl
                for i in range(Ml):
                    T4 += Z[i, be + 2] - Z[i, be + 1] + 1 / Nl
                grad[al * Nl + be + 1] = k * (
                            arrival_rates[al] * service_rates[be + 1] / pow((service_rates[be + 1] - T1), 2) - arrival_rates[al] * service_rates[be + 2] / pow(
                        (service_rates[be + 2] - T2), 2) \
                            + arrival_rates[al] * delta[al, be + 1] - arrival_rates[al] * delta[al, be + 2]) \
                                         - (1 / (Z[al, be + 1] - Z[al, be] + 1 / Nl) - 1 / (
                            Z[al, be + 2] - Z[al, be + 1] + 1 / Nl)) \
                                         - (-1 / (1 - (Z[al, be + 1] - Z[al, be] + 1 / Nl)) + 1 / (
                            1 - (Z[al, be + 2] - Z[al, be + 1] + 1 / Nl))) \
                                         - (-1 / (service_rates[be + 1] - T3) + 1 / (service_rates[be + 2] - T4))
                T1 *= 0
                T2 *= 0
                T3 *= 0
                T4 *= 0
        # be=Nl-2
        for al in range(Ml):
            for i in range(Ml):
                T1 += ((Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl) * arrival_rates[i])
            for i in range(Ml):
                T2 += ((Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i])
            for i in range(Ml):
                T3 += Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl
            for i in range(Ml):
                T4 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
            grad[al * Nl + Nl - 1] = k * (
                        arrival_rates[al] * service_rates[Nl - 1] / pow((service_rates[Nl - 1] - T1), 2) - arrival_rates[al] * service_rates[0] / pow(
                    (service_rates[0] - T2), 2) \
                        + arrival_rates[al] * delta[al, Nl - 1] - arrival_rates[al] * delta[al, 0]) \
                                     - (1 / (Z[al, Nl - 1] - Z[al, Nl - 2] + 1 / Nl) - 1 / (
                        Z[al, 0] - Z[al, Nl - 1] + 1 / Nl)) \
                                     - (-1 / (1 - (Z[al, Nl - 1] - Z[al, Nl - 2] + 1 / Nl)) + 1 / (
                        1 - (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl))) \
                                     - (-1 / (service_rates[Nl - 1] - T3) + 1 / (service_rates[0] - T4))
            T1 *= 0
            T2 *= 0
            T3 *= 0
            T4 *= 0
        # be=Nl-1
        for al in range(Ml):
            for i in range(Ml):
                T1 += ((Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i])
            for i in range(Ml):
                T2 += ((Z[i, 1] - Z[i, 0] + 1 / Nl) * arrival_rates[i])
            for i in range(Ml):
                T3 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
            for i in range(Ml):
                T4 += Z[i, 1] - Z[i, 0] + 1 / Nl
            grad[al * Nl] = k * (
                        arrival_rates[al] * service_rates[0] / pow((service_rates[0] - T1), 2) - arrival_rates[al] * service_rates[1] / pow((service_rates[1] - T2), 2) \
                        + arrival_rates[al] * delta[al, 0] - arrival_rates[al] * delta[al, 1]) \
                            - (1 / (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl) - 1 / (Z[al, 1] - Z[al, 0] + 1 / Nl)) \
                            - (-1 / (1 - (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl)) + 1 / (
                        1 - (Z[al, 1] - Z[al, 0] + 1 / Nl))) \
                            - (-1 / (service_rates[0] - T3) + 1 / (service_rates[1] - T4))
            T1 *= 0
            T2 *= 0
            T3 *= 0
            T4 *= 0
        return np.transpose(grad)

        # hessf

    def hessf(k, Z):
        # 대각성분 아닌거 먼저 만들고 자기자신의 transpose 더한 뒤에 대각 성분 넣기?
        hess = np.zeros((Ml * Nl, Ml * Nl))
        #
        R11 = 0
        R12 = 0
        for be in range(Nl - 1):
            for al in range(Ml):
                for i in range(Ml):
                    R11 += (Z[i, be + 1] - Z[i, be] + 1 / Nl) * arrival_rates[i]
                for i in range(Ml):
                    R12 += (Z[i, be + 1] - Z[i, be] + 1 / Nl)
                hess[Nl * al + be + 1, Nl * al + be] = k * (
                            -2 * pow(arrival_rates[al], 2) * service_rates[be + 1] / pow(service_rates[be + 1] - R11, 3)) \
                                                       - 1 / pow(Z[al, be + 1] - Z[al, be] + 1 / Nl, 2) - 1 / pow(
                    1 - (Z[al, be + 1] - Z[al, be] + 1 / Nl), 2) \
                                                       - 1 / pow(service_rates[be + 1] - R12, 2)
                R11 *= 0
                R12 *= 0
        for al in range(Ml):
            for i in range(Ml):
                R11 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
            for i in range(Ml):
                R12 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl)
            hess[Nl * al, Nl * al + Nl - 1] = k * (-2 * pow(arrival_rates[al], 2) * service_rates[0] / pow(service_rates[0] - R11, 3)) \
                                              - 1 / pow(Z[al, 0] - Z[al, Nl - 1] + 1 / Nl, 2) - 1 / pow(
                1 - (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl), 2) \
                                              - 1 / pow(service_rates[0] - R12, 2)
            R11 *= 0
            R12 *= 0
        #
        R21 = 0
        R22 = 0
        for be in range(Nl - 1):
            for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R21 += (Z[i, be + 1] - Z[i, be] + 1 / Nl) * arrival_rates[i]
                        for i in range(Ml):
                            R22 += Z[i, be + 1] - Z[i, be] + 1 / Nl
                        hess[Nl * al + be + 1, Nl * alp + be] = k * (
                                    -2 * service_rates[be + 1] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[be + 1] - R21, 3)) \
                                                                - 1 / pow(service_rates[be + 1] - R22, 2)
                        R21 *= 0
                        R22 *= 0
        for al in range(Ml):
            for alp in range(Ml):
                if al != alp:
                    for i in range(Ml):
                        R21 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
                    for i in range(Ml):
                        R22 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
                    hess[Nl * al, Nl * alp + Nl - 1] = k * (
                                -2 * service_rates[0] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[0] - R21, 3)) \
                                                       - 1 / pow(service_rates[0] - R22, 2)
                    R21 *= 0
                    R22 *= 0
        # traspose
        hess += np.transpose(hess)
        #
        R31 = 0
        R32 = 0
        R33 = 0
        R34 = 0
        for be in range(Nl - 2):
            for al in range(Ml):
                for alp in range(Ml):
                    if al != alp:
                        for i in range(Ml):
                            R31 += (Z[i, be + 1] - Z[i, be] + 1 / Nl) * arrival_rates[i]
                        for i in range(Ml):
                            R32 += (Z[i, be + 2] - Z[i, be + 1] + 1 / Nl) * arrival_rates[i]
                        for i in range(Ml):
                            R33 += (Z[i, be + 1] - Z[i, be] + 1 / Nl)
                        for i in range(Ml):
                            R34 += (Z[i, be + 2] - Z[i, be + 1] + 1 / Nl)
                        hess[Nl * al + be + 1, Nl * alp + be + 1] = k * (
                                    2 * service_rates[be + 1] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[be + 1] - R31, 3) + 2 * service_rates[
                                be + 2] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[be + 2] - R32, 3)) \
                                                                    + (1 / pow(service_rates[be + 1] - R33, 2) + 1 / pow(
                            service_rates[be + 2] - R34, 2))
                        R31 *= 0
                        R32 *= 0
                        R33 *= 0
                        R34 *= 0
        # be=Nl-2
        for al in range(Ml):
            for alp in range(Ml):
                if al != alp:
                    for i in range(Ml):
                        R31 += (Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl) * arrival_rates[i]
                    for i in range(Ml):
                        R32 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
                    for i in range(Ml):
                        R33 += (Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl)
                    for i in range(Ml):
                        R34 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl)
                    hess[Nl * al + Nl - 1, Nl * alp + Nl - 1] = k * (
                                2 * service_rates[Nl - 1] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[Nl - 1] - R31, 3) + 2 * service_rates[0] *
                                arrival_rates[al] * arrival_rates[alp] / pow(service_rates[0] - R32, 3)) \
                                                                + (1 / pow(service_rates[Nl - 1] - R33, 2) + 1 / pow(service_rates[0] - R34,
                                                                                                           2))
                    R31 *= 0
                    R32 *= 0
                    R33 *= 0
                    R34 *= 0
        # be=Nl-1
        for al in range(Ml):
            for alp in range(Ml):
                if al != alp:
                    for i in range(Ml):
                        R31 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
                    for i in range(Ml):
                        R32 += (Z[i, 1] - Z[i, 0] + 1 / Nl) * arrival_rates[i]
                    for i in range(Ml):
                        R33 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl)
                    for i in range(Ml):
                        R34 += (Z[i, 1] - Z[i, 0] + 1 / Nl)
                    hess[Nl * al, Nl * alp] = k * (
                                2 * service_rates[0] * arrival_rates[al] * arrival_rates[alp] / pow(service_rates[0] - R31, 3) + 2 * service_rates[1] * arrival_rates[
                            al] * arrival_rates[alp] / pow(service_rates[1] - R32, 3)) \
                                              + (1 / pow(service_rates[0] - R33, 2) + 1 / pow(service_rates[1] - R34, 2))
                    R31 *= 0
                    R32 *= 0
                    R33 *= 0
                    R34 *= 0
                    #
        R41 = 0
        R42 = 0
        R43 = 0
        R44 = 0
        for be in range(Nl - 2):
            for al in range(Ml):
                for i in range(Ml):
                    R41 += (Z[i, be + 1] - Z[i, be] + 1 / Nl) * arrival_rates[i]
                for i in range(Ml):
                    R42 += (Z[i, be + 2] - Z[i, be + 1] + 1 / Nl) * arrival_rates[i]
                for i in range(Ml):
                    R43 += Z[i, be + 1] - Z[i, be] + 1 / Nl
                for i in range(Ml):
                    R44 += Z[i, be + 2] - Z[i, be + 1] + 1 / Nl
                hess[Nl * al + be + 1, Nl * al + be + 1] = k * (
                            2 * pow(arrival_rates[al], 2) * service_rates[be + 1] / pow(service_rates[be + 1] - R41, 3) + 2 * pow(arrival_rates[al],
                                                                                                        2) * service_rates[
                                be + 2] / pow(service_rates[be + 2] - R42, 3)) \
                                                           + (1 / pow(Z[al, be + 1] - Z[al, be] + 1 / Nl, 2) + 1 / pow(
                    Z[al, be + 2] - Z[al, be + 1] + 1 / Nl, 2)) \
                                                           + (1 / pow(1 - (Z[al, be + 1] - Z[al, be] + 1 / Nl),
                                                                      2) + 1 / pow(
                    1 - (Z[al, be + 2] - Z[al, be + 1] + 1 / Nl), 2)) \
                                                           + (1 / pow(service_rates[be + 1] - R43, 2) + 1 / pow(service_rates[be + 2] - R44,
                                                                                                      2))
                R41 *= 0
                R42 *= 0
                R43 *= 0
                R44 *= 0
        # be=Nl-2
        for al in range(Ml):
            for i in range(Ml):
                R41 += (Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl) * arrival_rates[i]
            for i in range(Ml):
                R42 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
            for i in range(Ml):
                R43 += Z[i, Nl - 1] - Z[i, Nl - 2] + 1 / Nl
            for i in range(Ml):
                R44 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
            hess[Nl * al + Nl - 1, Nl * al + Nl - 1] = k * (
                        2 * pow(arrival_rates[al], 2) * service_rates[Nl - 1] / pow(service_rates[Nl - 1] - R41, 3) + 2 * pow(arrival_rates[al], 2) *
                        service_rates[0] / pow(service_rates[0] - R42, 3)) \
                                                       + (1 / pow(Z[al, Nl - 1] - Z[al, Nl - 2] + 1 / Nl, 2) + 1 / pow(
                Z[al, 0] - Z[al, Nl - 1] + 1 / Nl, 2)) \
                                                       + (1 / pow(1 - (Z[al, Nl - 1] - Z[al, Nl - 2] + 1 / Nl),
                                                                  2) + 1 / pow(1 - (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl),
                                                                               2)) \
                                                       + (1 / pow(service_rates[Nl - 1] - R43, 2) + 1 / pow(service_rates[0] - R44, 2))
            R41 *= 0
            R42 *= 0
            R43 *= 0
            R44 *= 0
        # be=Nl-1
        for al in range(Ml):
            for i in range(Ml):
                R41 += (Z[i, 0] - Z[i, Nl - 1] + 1 / Nl) * arrival_rates[i]
            for i in range(Ml):
                R42 += (Z[i, 1] - Z[i, 0] + 1 / Nl) * arrival_rates[i]
            for i in range(Ml):
                R43 += Z[i, 0] - Z[i, Nl - 1] + 1 / Nl
            for i in range(Ml):
                R44 += Z[i, 1] - Z[i, 0] + 1 / Nl
            hess[Nl * al, Nl * al] = k * (
                        2 * pow(arrival_rates[al], 2) * service_rates[0] / pow(service_rates[Nl - 1] - R41, 3) + 2 * pow(arrival_rates[al], 2) * service_rates[
                    1] / pow(service_rates[1] - R42, 3)) \
                                     + (1 / pow(Z[al, 0] - Z[al, Nl - 1] + 1 / Nl, 2) + 1 / pow(
                Z[al, 1] - Z[al, 0] + 1 / Nl, 2)) \
                                     + (1 / pow(1 - (Z[al, 0] - Z[al, Nl - 1] + 1 / Nl), 2) + 1 / pow(
                1 - (Z[al, 1] - Z[al, 0] + 1 / Nl), 2)) \
                                     + (1 / pow(service_rates[0] - R43, 2) + 1 / pow(service_rates[1] - R44, 2))
            R41 *= 0
            R42 *= 0
            R43 *= 0
            R44 *= 0
        return hess

        # barrier method

    while 1 / t > eps1:
        Nst = -np.matmul(np.linalg.inv(hessf(t, Y)), gradf(t, Y))
        Ndec = -np.matmul(np.transpose(gradf(t, Y)), Nst)
        while Ndec / 2 > eps2:
            s = 1
            while f(t, Y + s * np.reshape(Nst, (Ml, Nl))) > f(t, Y) + alpha * s * (-Ndec):
                s *= beta
            Y += s * np.reshape(Nst, (Ml, Nl))
            Nst = -np.matmul(np.linalg.inv(hessf(t, Y)), gradf(t, Y))
            Ndec = -np.matmul(np.transpose(gradf(t, Y)), Nst)
        t *= m
    # aij
    A = np.zeros((Ml, Nl))
    for i in range(Ml):
        A[i, :] = np.matmul(F, Y[i, :]) + 1 / Nl * np.transpose(np.ones(Nl))

        # lambda_hat
    lambda_hats = np.zeros(Nl)
    for j in range(Nl):
        for i in range(Ml):
            lambda_hats[j] += A[i, j] * arrival_rates[i]
    # service_time
    service_time = 0
    service1 = 0
    service2 = 0
    for i in range(Nl):
        service1 += lambda_hats[i] / (service_rates[i] - lambda_hats[i])
    for i in range(Ml):
        for j in range(Nl):
            service2 += arrival_rates[i] * delta[i, j] * A[i, j]
    service_time = service1 + service2
    # condition check
    D = np.zeros((Ml, Nl))
    for i in range(Ml):
        for j in range(Nl):
            D[i, j] = service_rates[j] / pow(service_rates[j] - lambda_hats[j], 2) + delta[i, j]
    service_time = service_time / sum(arrival_rates)
    result = {'A': A, 'lambda_hats': lambda_hats, 'Mean_completion_time': service_time, 'D': D}
    return result