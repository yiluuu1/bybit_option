import math
import cmath
from scipy import integrate
import pandas as pd


def cphiHN(phi, const):
    cphi0 = phi * complex(real=0, imag=1)
    cphi = cphi0 + const
    return cphi0, cphi


def fHN(cphi0, cphi, model, S, X, Time_inSteps, r_perStep, ht1=None):
    Lambda, omega, alpha, beta = -1 / 2, model['omega'], model['alpha'], model['beta']
    gamma = model['gamma'] + model['Lambda'] + 1 / 2
    sigma2 = (omega + alpha) / (1 - beta - alpha * gamma ** 2)
    if ht1 is not None:
        sigma2 = ht1
    a = cphi * r_perStep
    b = Lambda * cphi + cphi * cphi / 2
    for i in range(2, Time_inSteps + 1):
        a = a + cphi * r_perStep + b * omega - cmath.log(1 - 2 * alpha * b) / 2
        b = cphi * (Lambda + gamma) - gamma ** 2 / 2 + beta * b + 0.5 * (cphi - gamma) ** 2 / (1 - 2 * alpha * b)
    f = cmath.exp(-cphi0 * math.log(X) + cphi * math.log(S) + a + b * sigma2) / cphi0 / math.pi
    return f


def fpriceHN(phi, const, model, S, X, Time_inSteps, r_perStep, ht1):
    cphi0, cphi = cphiHN(phi, const)
    fprice = fHN(cphi0, cphi, model, S, X, Time_inSteps, r_perStep, ht1)
    return fprice.real


def fdeltaHN(phi, const, model, S, X, Time_inSteps, r_perStep):
    cphi0, cphi = cphiHN(phi, const)
    fdelta = cphi * fHN(cphi0, cphi, model, S, X, Time_inSteps, r_perStep) / S
    return fdelta.real


def fgammaHN(phi, const, model, S, X, Time_inSteps, r_perStep):
    cphi0, cphi = cphiHN(phi, const)
    fgamma = cphi * (cphi - 1) * fHN(cphi0, cphi, model, S, X, Time_inSteps, r_perStep) / S ** 2
    return fgamma.real


def HNGOptionDelta(TypeFlag, model, S, X, Time_inSteps, r_perStep, ht1):
    call1 = integrate.quad(fpriceHN, 0, 10000, args=(1, model, S, X, Time_inSteps, r_perStep, ht1))
    if TypeFlag == "c":
        d = math.exp(-r_perStep * Time_inSteps) * call1[0] / S + 1 / 2
    else:
        d = math.exp(-r_perStep * Time_inSteps) * call1[0] / S - 1 / 2
    return d


def HNGDelta_D(TypeFlag, model, S, X, Time_inSteps, r_perStep, ht1=None):
    d = HNGOptionDelta(TypeFlag, model, S, X, Time_inSteps, r_perStep, ht1)
    return d


def HNGGamma_D(TypeFlag, model, S, X, Time_inSteps, r_perStep, ht1=None):
    sigma = math.sqrt((model['alpha'] + model['omega']) / (1 - model['beta'] - model['alpha'] * model['gamma'] ** 2))
    S1, S2 = S + sigma, S - sigma
    d1 = HNGOptionDelta(TypeFlag, model, S1, X, Time_inSteps, r_perStep, ht1)
    d2 = HNGOptionDelta(TypeFlag, model, S2, X, Time_inSteps, r_perStep, ht1)
    greek = pd.Series({'Delta': (d1 + d2) / 2, 'Gamma': (d1 - d2) / (2 * sigma)}, name='Greek')
    return greek


def HNGOption(TypeFlag, model, S, X, Time_inSteps, r_perStep, ht1=None):
    call1 = integrate.quad(fpriceHN, 0, 10000, args=(1, model, S, X, Time_inSteps, r_perStep, ht1))
    call2 = integrate.quad(fpriceHN, 0, 10000, args=(0, model, S, X, Time_inSteps, r_perStep, ht1))
    call_price = S / 2 + math.exp(-r_perStep * Time_inSteps) * call1[0] - X * math.exp(-r_perStep * Time_inSteps) * (
                1 / 2 + call2[0])
    price = None
    if TypeFlag == "c":
        price = call_price
    if TypeFlag == "p":
        price = call_price + X * math.exp(-r_perStep * Time_inSteps) - S
    return price


def HNGGreeks(Selection, TypeFlag, model, S, X, Time_inSteps, r_perStep):
    if Selection == "Delta":
        delta1 = integrate.quad(fdeltaHN, 0, 10000, args=(1, model, S, X, Time_inSteps, r_perStep))
        delta2 = integrate.quad(fdeltaHN, 0, 10000, args=(0, model, S, X, Time_inSteps, r_perStep))
        if TypeFlag == "c":
            greek = 1 / 2 + math.exp(-r_perStep * Time_inSteps) * delta1[0] \
                    - X * math.exp(-r_perStep * Time_inSteps) * delta2[0]
        else:
            greek = -1 / 2 + math.exp(-r_perStep * Time_inSteps) * delta1[0] \
                    - X * math.exp(-r_perStep * Time_inSteps)* delta2[0]
    else:
        gamma1 = integrate.quad(fgammaHN, 0, 10000, args=(1, model, S, X, Time_inSteps, r_perStep))
        gamma2 = integrate.quad(fgammaHN, 0, 10000, args=(0, model, S, X, Time_inSteps, r_perStep))
        greek = math.exp(-r_perStep * Time_inSteps) * gamma1[0] - X * math.exp(-r_perStep * Time_inSteps) * gamma2[0]
    return greek
