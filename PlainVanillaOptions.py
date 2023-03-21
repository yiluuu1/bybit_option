import numpy as np
import pandas as pd
from datetime import *
from scipy import interpolate, stats


#吴老师自己从头开始写的
def NDF(x):
    result = np.exp(-x * x / 2) / np.sqrt(8 * np.arctan(1))
    return result


def CND(x):
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    k = 1 / (1 + 0.2316419 * abs(x))
    result = NDF(x) * (a1 * k + a2 * k ** 2 + a3 * k ** 3 + a4 * k ** 4 + a5 * k ** 5) - 0.5
    result = 0.5 - result * np.sign(x)
    return result


def CBND(x1, x2, rho):
    a, b = x1, x2
    if abs(rho) == 1:
        rho = rho - 1e-12 * np.sign(rho)
    X = [0.24840615, 0.39233107, 0.21141819, 0.03324666, 0.00082485334]
    Y = [0.10024215, 0.48281397, 1.0609498, 1.7797294, 2.6697604]
    a1, b1 = a / np.sqrt(2 * (1 - rho ** 2)), b / np.sqrt(2 * (1 - rho ** 2))
    if a <= 0 and b <= 0 and rho <= 0:
        Sum1 = 0
        for I in range(5):
            for J in range(5):
                Sum1 += X[I] * X[J] * np.exp(
                    a1 * (2 * Y[I] - a1) + b1 * (2 * Y[J] - b1)
                    + 2 * rho * (Y[I] - a1) * (Y[J] - b1))
        result = np.sqrt(1 - rho ** 2) / np.pi * Sum1
        return result
    elif a <= 0 and b >= 0 and rho >= 0:
        result = CND(a) - CBND(a, -b, -rho)
        return result
    elif a >= 0 and b <= 0 and rho >= 0:
        result = CND(b) - CBND(-a, b, -rho)
        return result
    elif a >= 0 and b >= 0 and rho <= 0:
        result = CND(a) + CND(b) - 1 + CBND(-a, -b, rho)
        return result
    elif a * b * rho >= 0:
        rho1 = (rho * a - b) * np.sign(a) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        rho2 = (rho * b - a) * np.sign(b) / np.sqrt(a ** 2 - 2 * rho * a * b + b ** 2)
        delta = (1 - np.sign(a) * np.sign(b)) / 4
        result = CBND(a, 0, rho1) + CBND(b, 0, rho2) - delta
        return result


def GBSOption(TypeFlag, S, X, Time, r, b, sigma, title=None, description=None):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - sigma * np.sqrt(Time)
    if TypeFlag == "c":
        result = S * np.exp((b - r) * Time) * CND(d1) - X * np.exp(-r * Time) * CND(d2)
    else:
        result = X * np.exp(-r * Time) * CND(-d2) - S * np.exp((b - r) * Time) * CND(-d1)
    param = pd.Series({'TypeFlag': TypeFlag, 'S': S, 'X': X,
                       'Time': Time, 'r': r, 'b': b, 'sigma': sigma}, name='param')
    if title is None:
        title = "Black Scholes Option Valuation"
    if description is None:
        description = str(datetime.now())
    fOPTION = pd.Series({'param': param, 'price': result,
                         'title': title, 'description': description}, name='fOPTION')
    return fOPTION


def GBSGreeks(Selection, TypeFlag, S, X, Time, r, b, sigma):
    result = None
    if Selection == "Delta" or Selection == "delta":
        result = GBSDelta(TypeFlag, S, X, Time, r, b, sigma)
    elif Selection == "Theta" or Selection == "theta":
        result = GBSTheta(TypeFlag, S, X, Time, r, b, sigma)
    elif Selection == "Vega" or Selection == "vega":
        result = GBSVega(S, X, Time, r, b, sigma)
    elif Selection == "Rho" or Selection == "rho":
        result = GBSRho(TypeFlag, S, X, Time, r, b, sigma)
    elif Selection == "Lambda" or Selection == "lambda":
        result = GBSLambda(TypeFlag, S, X, Time, r, b, sigma)
    elif Selection == "Gamma" or Selection == "gamma":
        result = GBSGamma(S, X, Time, r, b, sigma)
    elif Selection == "CofC" or Selection == "cofc":
        result = GBSCofC(TypeFlag, S, X, Time, r, b, sigma)
    return result


def GBSDelta(TypeFlag, S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    if TypeFlag == "c":
        result = np.exp((b - r) * Time) * CND(d1)
    else:
        result = np.exp((b - r) * Time) * (CND(d1) - 1)
    return result


def GBSTheta(TypeFlag, S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - sigma * np.sqrt(Time)
    Theta1 = -(S * np.exp((b - r) * Time) * NDF(d1) * sigma) / (2 * np.sqrt(Time))
    if TypeFlag == "c":
        result = Theta1 - (b - r) * S * np.exp((b - r) * Time) * CND(+d1) - r * X * np.exp(-r * Time) * CND(+d2)
    else:
        result = Theta1 + (b - r) * S * np.exp((b - r) * Time) * CND(-d1) + r * X * np.exp(-r * Time) * CND(-d2)
    return result


def GBSVega(S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    result = S * np.exp((b - r) * Time) * NDF(d1) * np.sqrt(Time)
    return result


def GBSRho(TypeFlag, S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    d2 = d1 - sigma * np.sqrt(Time)
    CallPut = GBSOption(TypeFlag, S, X, Time, r, b, sigma)['price']
    if TypeFlag == "c":
        if b != 0:
            result = Time * X * np.exp(-r * Time) * CND(d2)
        else:
            result = -Time * CallPut
    else:
        if b != 0:
            result = -Time * X * np.exp(-r * Time) * CND(-d2)
        else:
            result = -Time * CallPut
    return result


def GBSLambda(TypeFlag, S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    CallPut = GBSOption(TypeFlag, S, X, Time, r, b, sigma)['price']
    if TypeFlag == "c":
        result = np.exp((b - r) * Time) * CND(d1) * S / CallPut
    else:
        result = np.exp((b - r) * Time) * (CND(d1) - 1) * S / CallPut
    return result


def GBSGamma(S, X, Time, r, b, sigma):

    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    result = np.exp((b - r) * Time) * NDF(d1) / (S * sigma * np.sqrt(Time))
    return result


def GBSCofC(TypeFlag, S, X, Time, r, b, sigma):
    d1 = (np.log(S / X) + (b + sigma * sigma / 2) * Time) / (sigma * np.sqrt(Time))
    if TypeFlag == "c":
        result = Time * S * np.exp((b - r) * Time) * CND(d1)
    else:
        result = -Time * S * np.exp((b - r) * Time) * CND(-d1)
    return result


def GBSCharacteristics(TypeFlag, S, X, Time, r, b, sigma):
    premium = GBSOption(TypeFlag, S, X, Time, r, b, sigma)['price']
    delta = GBSGreeks("Delta", TypeFlag, S, X, Time, r, b, sigma)
    theta = GBSGreeks("Theta", TypeFlag, S, X, Time, r, b, sigma)
    vega = GBSGreeks("Vega", TypeFlag, S, X, Time, r, b, sigma)
    rho = GBSGreeks("Rho", TypeFlag, S, X, Time, r, b, sigma)
    Lambda = GBSGreeks("Lambda", TypeFlag, S, X, Time, r, b, sigma)
    gamma = GBSGreeks("Gamma", TypeFlag, S, X, Time, r, b, sigma)
    Characteristics = pd.Series({'premium': premium, 'delta': delta, 'theta': theta,
                                 'vega': vega, 'rho': rho, 'Lambda': Lambda, 'gamma': gamma},
                                name='Characteristics')
    return Characteristics


def BlackScholesOption(TypeFlag, S, X, Time, r, b, sigma, title=None, description=None):
    GBSOption(TypeFlag, S, X, Time, r, b, sigma, title, description)


def Black76Option(TypeFlag, FT, X, Time, r, sigma, title=None, description=None):
    result = GBSOption(TypeFlag=TypeFlag, S=FT, X=X, Time=Time,
                       r=r, b=0, sigma=sigma)['price']
    param = pd.Series({'TypeFlag': TypeFlag, 'FT': FT, 'X': X,
                       'Time': Time, 'r': r, 'sigma': sigma}, name='param')
    if title is None:
        title = "Black 76 Option Valuation"
    if description is None:
        description = str(datetime.now())
    fOPTION = pd.Series({'param': param, 'price': result,
                         'title': title, 'description': description}, name='fOPTION')
    return fOPTION


def GBSVolatility(price, TypeFlag, S, X, Time, r, b):
    # max = 10000
    # precsion = 0.00001
    # sigma = 0.5
    # for i in range(max):
    #     c_est = GBSOption(TypeFlag=TypeFlag, S=S, X=X, Time=Time,
    #                       r=r, b=b, sigma=sigma)['price']
    #     vega = GBSVega(S=S, X=X, Time=Time, r=r, b=b, sigma=sigma)
    #     diff = price - c_est
    #     if abs(diff) < precsion:
    #         return sigma
    #     sigma = sigma + diff/vega
    # return sigma
    c_est = 0
    top, floor= 10, 0
    sigma = 0.5
    count = 0
    while abs(price - c_est) > 0.0001:
        c_est = GBSOption(TypeFlag=TypeFlag, S=S, X=X, Time=Time,
                          r=r, b=b, sigma=sigma)['price']
        count += 1
        if count > 10000:
            sigma = 0
            break
        if price - c_est > 0:
            floor = sigma
        else:
            top = sigma
        sigma = (top + floor) / 2
    return sigma


##BSM基础计算公式
##Numpy完成的基础部分
def bsm_price(sigma, typeflag, underlying, strike, t2exp, r):
    d1 = (np.log(underlying / strike) + (r + sigma * sigma / 2) * t2exp)\
         / (sigma * np.sqrt(t2exp))
    d2 = (np.log(underlying / strike) + (r - sigma * sigma / 2) * t2exp)\
         / (sigma * np.sqrt(t2exp))
    if typeflag == "c":
        price = underlying * stats.norm.cdf(d1) \
                - strike * np.exp(-r * t2exp) * stats.norm.cdf(d2)
    else:
        price = strike * np.exp(-r * t2exp) * stats.norm.cdf(-d2) \
                - underlying * stats.norm.cdf(-d1)
    return price


def bsm_volatility(price, typeflag, underlying, strike, t2exp, r):
    # 包含二分法/积分法
    # 当前方法为：二分法
    sigma = 5
    top, floor = 100, 0
    for i in range(10000):
        c_est = bsm_price(sigma, typeflag, underlying, strike, t2exp, r)
        # vega = bsm_vega(sigma, underlying, strike, t2exp, r)
        diff = price - c_est
        if abs(diff) < 0.00001:
            return sigma
        else:
            # sigma = sigma + diff / vega
            if diff > 0:
                floor = sigma
            else:
                top = sigma
            sigma = (top + floor) / 2
    return sigma


def bsm_delta(sigma, typeflag, underlying, strike, t2exp, r):
    d1 = (np.log(underlying / strike) + (r + sigma * sigma / 2) * t2exp)\
         / (sigma * np.sqrt(t2exp))
    if typeflag == "c":
        delta = np.exp(-r * t2exp) * stats.norm.cdf(d1)
    else:
        delta = np.exp(-r * t2exp) * (stats.norm.cdf(d1) - 1)
    return delta


def bsm_gamma(sigma, underlying, strike, t2exp, r):
    d1 = (np.log(underlying / strike) + (r + sigma * sigma / 2) * t2exp)\
         / (sigma * np.sqrt(t2exp))
    gamma = (np.exp(-r * t2exp) * stats.norm.pdf(d1) /
             (underlying * sigma * np.sqrt(t2exp)))
    return gamma


def bsm_vega(sigma, underlying, strike, t2exp, r):
    d1 = (np.log(underlying / strike) + (r + sigma * sigma / 2) * t2exp)\
         / (sigma * np.sqrt(t2exp))
    vega = underlying * np.exp(-r * t2exp) * stats.norm.pdf(d1) * np.sqrt(t2exp)
    return vega
