import numpy as np
from scipy.stats import norm


def produce_xi(x1, x2):
    while True:
        xi = np.random.normal(size=1)
        if x1 < xi < x2:
            break
    return xi


def produce_param(param, sigma, mL, mH):
    ext = 10
    mL = min(param, mL)
    mH = max(param, mH)
    sigma = min(sigma, (mH - mL) / ext)
    ximin = ext * (mL - param) / (mH - mL)
    ximax = ext * (mH - param) / (mH - mL)
    if ximin > 2 or ximax < -2:
        return np.random.uniform(low=ximin, high=ximax, size=1) * sigma + param
    else:
        return param + sigma * produce_xi(ximin, ximax)


def llhaff(param, rfr, x, final=False):
    Z = np.zeros(shape=len(x))
    q = np.zeros(shape=len(x))
    h = np.zeros(shape=len(x))
    lamda = param[0]
    omega = param[1]
    alpha = param[2]
    beta = param[3]
    phi = param[4]
    rho = param[5]
    gam1 = param[6]
    gam2 = param[7]

    h[0] = omega / (1 - rho)
    q[0] = omega / (1 - rho)
    Z[0] = (x[0] - rfr - lamda * h[0]) / np.sqrt(h[0])
    for i in range(1, len(Z)):
        q[i] = omega + rho * q[i - 1] + phi * h[i - 1] * (Z[i - 1] ** 2 - 2 * gam2 * Z[i - 1] - 1)
        h[i] = q[i] + beta * (h[i - 1] - q[i - 1]) + alpha * h[i - 1] * (Z[i - 1] ** 2 - 2 * gam1 * Z[i - 1] - 1)
        Z[i] = (x[i] - rfr - lamda * h[i]) / np.sqrt(h[i])

    llhaff = sum(np.log(norm.pdf(Z) / np.sqrt(h)))
    if final:
        return [llhaff, Z, h, q]
    else:
        return llhaff


def ls_generatepar_nextstep(bsL, bsH, sigma, param0, j, rfr, x):
    t = 5 / (24 * 60 * 365)
    minq = 0.2 ** 2 * t
    maxq = 2.5 ** 2 * t
    sat = False
    param = param0
    time = 0
    while ~sat and time <= 30:
        mi = bsL[j]
        ma = bsH[j]

        if j == 1:
            rho = param[5]
            mi = max(mi, (1 - rho) * minq)
            ma = min(ma, (1 - rho) * maxq)
        # lamda = theta[0]
        if j == 5:
            omega = param[1]
            mi = max(mi, (minq - omega) / minq)
            ma = min(ma, (maxq - omega) / maxq)
        param[j] = produce_param(param0[j], sigma[j], mi, ma)
        omega = param[1]
        rho = param[5]
        q0 = omega / (1 - rho)
        if q0 > maxq or q0 < minq:
            continue

        llh = llhaff(param=param, rfr=rfr, x=x)
        sat = ~np.isnan(llh) and ~np.isinf(llh)

    return [param, llh, time >= 29]


def ls_generatepar(bsL, bsH, rfr, x):
    t = 5 / (24 * 60 * 365)
    minq = 0.2 ** 2 * t
    maxq = 2.5 ** 2 * t
    sat = False
    param = np.zeros(len(bsL))
    while ~sat:
        for j in range(len(bsL)):
            param[j] = np.random.uniform(low=bsL[j], high=bsH[j], size=1)
        rho = param[5]
        param[1] = np.random.uniform(low=(1 - rho) * minq, high=(1 - rho) * maxq, size=1)
        omega = param[1]

        q0 = omega / (1 - rho)
        if q0 > maxq or q0 < minq:
            continue
        llh = llhaff(param=param, rfr=rfr, x=x)
        sat = ~np.isnan(llh) and ~np.isinf(llh)

    return [param, llh]


def lsngarch_bayes(x):
    bsL = np.array([-.002, 0, 0.25, 0, 0, 0.9, 0, 0])
    bsH = np.array([.002, .01, 1, 0.8, 1, 1, 1, 1])
    sigma = bsH - bsL
    sigma0 = sigma
    rfr = 0
    re = ls_generatepar(bsL, bsH, rfr, x)
    param = re[0]
    param[0] = 0
    llh1 = re[1]

    M = 2000
    mat1 = np.zeros((M - 1000, len(param) + 1))
    mat1[0, :len(param)] = param
    mat1[0, len(param)] = llh1

    for i in range(1, M):
        for j in range(1, len(param)):
            param0 = param
            re = ls_generatepar_nextstep(bsL, bsH, sigma, param0, j, rfr, x)
            param = re[0]
            llh2 = re[1]
            stop = re[2]
            u = np.random.uniform(size=1)
            acceptprob = min(np.exp(llh2 - llh1), 1)
            if u > acceptprob or stop:
                param = param0
            else:
                llh1 = llh2
            print((i - 1) * (len(param)) + j)
        if i >= 1000:
            mat1[i - 1000, :len(param)] = param
            mat1[i - 1000, len(param)] = llh1
            if i >= 1500:
                m2 = mat1[(i - 1200):(i - 1000 + 1), :-1].std(axis=0)
                for nn in range(1, len(sigma)):
                    sigma[nn] = max(sigma0[nn] / 100, 2 * m2[nn])

    param = mat1[(M - 1101):(M - 1000), :-1].mean(axis=0)

    lamda = param[0]
    omega = param[1]
    alpha = param[2]
    beta = param[3]
    phi = param[4]
    rho = param[5]
    gam1 = param[6]
    gam2 = param[7]

    class Model:
        def __init__(self, lamda, omega, alpha, beta, phi, rho, gam1, gam2, rfr):
            self.lamda = lamda
            self.omega = omega
            self.alpha = alpha
            self.beta = beta
            self.phi = phi
            self.rho = rho
            self.gamma1 = gam1
            self.gamma2 = gam2
            self.rfr = rfr

    model = Model(lamda, omega, alpha, beta, phi, rho, gam1, gam2, rfr)
    estimate = {'lamda': lamda, 'omega': omega, 'alpha': alpha, 'beta': beta, 'phi': phi, 'rho': rho, 'gam1': gam1,
                'gam2': gam2, 'rfr': rfr}

    llh, Z, h, q = llhaff(param, rfr, x, final=True)

    # Statistics - Printing:
    sigma2 = omega / (1 - rho)

    # Return Value:
    class opt:
        def __init__(self, model, estimate, llh, Z, h, q, x, sigma2):
            self.model = model
            self.estimate = estimate
            self.llh = llh
            self.z = Z
            self.h = h
            self.q = q
            self.x = x
            self.sigma2 = sigma2

    hngarch = opt(model, estimate, llh, Z, h, q, x, sigma2)
    return hngarch
