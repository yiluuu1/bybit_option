import numpy as np
import pandas as pd
from PlainVanillaOptions import GBSGreeks


# def ajgarchSim(model, ht1=None, nt=1000):
#     Lambda = model.lamda
#     omega = model.omega
#     alpha = model.alpha
#     beta = model.beta
#     #kappa = model['kappa']
#     gamma = model.gamma
#     #theta = model['theta']
#     #delta = model['delta']
#     rfr = model.rfr
#
#     x = h = Z = y = np.zeros(nt)
#
#     if ht1 is None:
#         h[0] = (omega+alpha) / (1-alpha*gamma*gamma-beta)
#     else:
#         h[0] = ht1
#     for i in range(nt):
#         Z[i] = np.random.normal(1)
#         ny = np.random.poisson(kappa*h[i],1)
#         y[i] = 0
#         if ny > 0:
#             for n in range(ny):
#                 y[i] += np.random.normal(loc=theta, scale=math.sqrt(delta), size=1)
#         x[i] = rfr+(Lambda-0.5)*h[i]+math.sqrt(h[i])*Z[i]+y[i]
#         if i < nt:
#             h[i+1] = omega+alpha*(Z[i]-gamma*math.sqrt(h[i]))**2+beta*h[i]
#     return x
def JNPath(model, empZ, S0, L, ht1=None):
    x = agarchSim(model=model, empZ=empZ, ht1=ht1, UnderType='no', n=L)
    S = S0 * np.exp(np.cumsum(x))
    return S


def agarchSim(model, empZ, UnderType, ht1=None, n=1000):
    innov = np.random.choice(empZ, n)
    Lambda = model['Lambda']
    omega = model['omega']
    alpha = model['alpha']
    beta = model['beta']
    gamma = model['gamma']
    rfr = model['rf']
    x = h = Z = innov
    nt = n

    if ht1 is None:
        h[1] = (omega + alpha) / (1 - alpha * gamma * gamma - beta)
    else:
        h[1] = ht1
    if UnderType == "no":
        Z = np.random.normal(size=nt)
    if UnderType == "t":
        nv = model['theta']
        Z = np.random.standard_t(size=nt, df=nv) * np.sqrt((nv - 2) / nv)
    for i in range(nt):
        # x[i] = rfr-0.5*h[i]+math.sqrt(h[i])*Z[i]
        x[i] = rfr + Lambda * h[i] + np.sqrt(h[i]) * Z[i]
        if i < nt:
            h[i + 1] = omega + alpha * (Z[i] - gamma * np.sqrt(h[i])) ** 2 + beta * h[i]
    return x


def ngarchSim(model, empZ, UnderType, ht1=None, n=1000):
    innov = np.random.normal(n)
    if UnderType == 'hn':
        innov = np.random.choice(empZ, n)
    Lambda = model.lamda
    omega = model.omega
    alpha = model.alpha
    beta = model.beta
    gamma = model.gam
    rfr = model.rfr
    x = innov
    h = innov
    Z = innov
    nt = n

    if ht1 is None:
        h[0] = omega / (1 - alpha * (1 + gamma ** 2) - beta)
    else:
        h[0] = ht1
    if UnderType == "no":
        Z = np.random.normal(size=nt)
    for i in range(nt):
        x[i] = rfr + Lambda * h[i] + np.sqrt(h[i]) * Z[i]
        if i < nt - 1:
            h[i + 1] = omega + alpha * h[i] * (Z[i] - gamma) ** 2 + beta * h[i]

    return x


def ls_ngarchSim(model, empZ, UnderType, ht1=None, qt1=None, n=1000):
    Z = np.random.normal(n)
    if UnderType == "hn":
        Z = np.random.choice(empZ, size=n)
    lamda = 0
    omega = model.omega
    alpha = model.alpha
    beta = model.beta
    gamma1 = model.gamma1
    gamma2 = model.gamma2
    rfr = model.rfr
    rho = model.rho
    phi = model.phi
    x = np.zeros(n)
    h = np.zeros(n)
    q = np.zeros(n)

    if np.isnan(ht1):
        h[0] = omega / (1 - rho)
    else:
        h[0] = ht1
    if np.isnan(qt1):
        q[0] = omega / (1 - rho)
    else:
        q[0] = qt1
    # if UnderType == "no":
    #   Z <- rnorm(n = nt)
    # if UnderType == "t":
    #   nv <- model$theta
    #   Z <- rt(n=nt,df=nv)*sqrt((nv-2)/nv)
    for i in range(n):
        x[i] = rfr + lamda * h[i] + np.sqrt(h[i]) * Z[i]
        if i < n - 1:
            q[i + 1] = omega + rho * q[i] + phi * h[i] * (Z[i] ** 2 - 2 * gamma2 * Z[i] - 1)
            h[i + 1] = q[i + 1] + beta * (h[i] - q[i]) + alpha * h[i] * (Z[i] ** 2 - 2 * gamma1 * Z[i] - 1)
    return x


def HestonNandiPath(model, empZ, S0, L, UnderType, GarchType, ht1=None):
    while True:
        if GarchType == "agarch":
            x = agarchSim(model=model, empZ=empZ, UnderType=UnderType, n=L, ht1=ht1)
        elif GarchType == 'ngarch':
            x = ngarchSim(model=model, empZ=empZ, UnderType=UnderType, n=L, ht1=ht1)
        elif GarchType == 'lsngarch':
            x = ls_ngarchSim(model, empZ, UnderType, ht1=ht1[0], qt1=ht1[1], n=L)
        # elif GarchType=='lslamngarch':
        #     x = ls_lamngarchSim(model=model, empZ=empZ, UnderType=UnderType, n=L, ht1=ht1)
        S = np.append(S0, S0 * np.exp(np.cumsum(x)))
        # if sum(np.isinf(S)) or sum(np.isnan(S)) or sum(np.isnan(S)):
        #     continue
        if max(S) < S0 * 2 or min(S) > S0 / 2:
            break
    return S


def generatePath(model, empZ, L, ht1=None, iteration=1000,
                 UnderType="hn", GarchType="agarch"):
    Spath = pd.DataFrame()
    for n in range(iteration):
        if UnderType == "jn":
            # S = JNPath(model, empZ, S0, ht1)
            pass
        else:
            S = HestonNandiPath(model, empZ, 1, L, UnderType, GarchType, ht1)
        Spath = Spath.append(pd.DataFrame([S]))
    return Spath


def hedgeMCPriceall(S0, K, model, L, Spath, TypeFlag='c', ht1=None, GarchType='agarch'):
    # if GarchType == "agarch":
    #     sigma_perStep = np.sqrt((model.omega+model.alpha) / (1 - model.beta-model.alpha * gamma ** 2))
    # elif GarchType == "ngarch":
    #     sigma_perStep =np.sqrt((model.omega) / (1 - model.beta-model.alpha * (1+gamma ** 2)))
    if GarchType == "lsngarch":
        sigma_perStep = np.sqrt(model.omega / (1 - model.rho))
    if ~np.isnan(ht1[0]):
        sigma_perStep = np.sqrt(ht1[0])
    r_perStep = model.rfr
    iteration = len(Spath)
    cb = np.zeros(iteration)
    for n in range(iteration):
        S = Spath.iloc[n, :]
        S = S0 * S[:L + 1]
        nT = len(S)
        Gamma = np.zeros(nT - 1)
        Delta = np.zeros(nT - 1)
        for ii in range(nT - 1):
            t = nT - ii - 1
            Gamma[ii] = GBSGreeks('Gamma', TypeFlag=TypeFlag, S=S[ii], X=K,
                                  Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
            Delta[ii] = GBSGreeks("Delta", TypeFlag=TypeFlag, S=S[ii], X=K,
                                  Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
        reb = opt_replicate(S, K, Delta, Gamma, r_perStep, TypeFlag, longOpt=1)
        cb[n] = reb[0] / (1 + r_perStep) ** nT
    return cb


def hedgeMCPrice(S0, K, model, empZ, L, TypeFlag='c', ht1=None, iteration=1000,
                 UnderType='hn', GarchType='agarch'):
    omega = model.omega
    alpha = model.alpha
    beta = model.beta
    gamma = model.gam
    rfr = model.rfr

    if GarchType == "agarch":
        sigma_perStep = np.sqrt((omega + alpha) / (1 - beta - alpha * gamma ** 2))
    else:
        sigma_perStep = np.sqrt(omega / (1 - beta - alpha * (1 + gamma ** 2)))
    if ht1 is not None:
        sigma_perStep = np.sqrt(ht1)
    r_perStep = rfr
    cb = np.zeros(iteration)
    for n in range(iteration):
        if UnderType == "jn":
            S = JNPath(model, empZ, S0, L, ht1)
        else:
            S = HestonNandiPath(model, empZ, S0, L, UnderType, GarchType, ht1)
        nT = len(S)
        Gamma = np.zeros(nT - 1)
        Delta = np.zeros(nT - 1)
        for ii in range(nT - 1):
            t = nT - ii - 1
            Gamma[ii] = GBSGreeks("Gamma", TypeFlag=TypeFlag, S=S[ii], X=K,
                                  Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
            Delta[ii] = GBSGreeks("Delta", TypeFlag=TypeFlag, S=S[ii], X=K,
                                  Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
        reb = opt_replicate(S, K, Delta, Gamma, r_perStep, longOpt=1, TypeFlag=TypeFlag)
        cb[n] = reb[1] / (1 + r_perStep) ** nT
    return cb


def hedgePrice(S, K, r_perStep, sigma_perStep, TypeFlag='c'):
    nT = len(S)
    Gamma = np.zeros(nT - 1)
    Delta = np.zeros(nT - 1)
    for ii in range(nT - 1):
        t = nT - ii - 1
        Gamma[ii] = GBSGreeks("Gamma", TypeFlag="c", S=S[ii], X=K,
                              Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
        Delta[ii] = GBSGreeks("Delta", TypeFlag=TypeFlag, S=S[ii], X=K,
                              Time=t, r=r_perStep, b=r_perStep, sigma=sigma_perStep)
    reb = opt_replicate(S, K, Delta, Gamma, r_perStep, longOpt=1, TypeFlag=TypeFlag)
    rea = opt_replicate(S, K, Delta, Gamma, r_perStep, longOpt=0, TypeFlag=TypeFlag)
    c_b = reb[0] / (1 + r_perStep) ** nT - reb[1]
    c_a = -rea[0] / (1 + r_perStep) ** nT + rea[1]
    return [c_b, c_a]


def opt_replicate(S, K, Delta, Gamma, r_perStep, TypeFlag, longOpt):
    nT = len(S)
    X = 0
    cost = 0
    cd = 0
    for ii in range(1, nT):
        t = nT - ii - 1
        band = max(1e-3, (3 / 2 * np.exp(-r_perStep * t) * 5e-4 * S[ii - 1] * Gamma[ii - 1] ** 2) ** (1.0 / 3))
        band = min(band, 1e-1) *0
        if abs((1 - 2 * longOpt) * Delta[ii - 1] - cd) > band:
            # M = abs((1 - 2 * longOpt) * Delta[ii - 1] - cd)
            transaction = abs((1 - 2 * longOpt) * Delta[ii - 1] - cd) * 5e-4
            cd = (1 - 2 * longOpt) * Delta[ii - 1] + band * 0.5 * np.sign(cd - (1 - 2 * longOpt) * Delta[ii - 1])
            cost += transaction
        X = cd * S[ii] + (1 + r_perStep) * (X - cd * S[ii - 1])

    if TypeFlag == "c":
        X = X - (1 - 2 * longOpt) * max(S[nT - 1] - K, 0)
    else:
        X = X - (1 - 2 * longOpt) * max(K - S[nT - 1], 0)
    #  print(X)
    return [X, cost]
