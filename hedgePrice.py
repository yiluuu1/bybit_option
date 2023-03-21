import numpy as np
import pandas as pd
from garch.Basic_Option_Func import BSMoption


def ls_ngarchSim(model, empZ, ht1=None, qt1=None, n=1000):
    innov = np.random.choice(empZ, size=n)
    Z = innov.copy()
    x = innov.copy()
    h = np.zeros(n + 1)
    q = np.zeros(n + 1)

    lamda = 0
    omega = model.omega
    alpha = model.alpha
    beta = model.beta
    gamma1 = model.gamma1
    gamma2 = model.gamma2
    rfr = model.rfr
    rho = model.rho
    phi = model.phi

    if np.isnan(ht1):
        h[0] = omega / (1 - rho)
    else:
        h[0] = ht1
    if np.isnan(qt1):
        q[0] = omega / (1 - rho)
    else:
        q[0] = qt1
    for i in range(n):
        x[i] = rfr + lamda * h[i] + np.sqrt(h[i]) * Z[i]
        if i < n - 1:
            q[i + 1] = omega + rho * q[i] + phi * h[i] * (Z[i] ** 2 - 2 * gamma2 * Z[i] - 1)
            h[i + 1] = q[i + 1] + beta * (h[i] - q[i]) + alpha * h[i] * (Z[i] ** 2 - 2 * gamma1 * Z[i] - 1)
    return x, h, q


def HestonNandiPath(model, empZ, S0, L, GarchType='lsngarch', ht1=None):
    while True:
        if GarchType == 'lsngarch':
            x, h, q = ls_ngarchSim(model, empZ, ht1=ht1[0], qt1=ht1[1], n=L)
        else:
            x, h, q = None, None, None
        S = np.append(S0, S0 * np.exp(np.cumsum(x)))
        if np.isnan(S).any() or np.isinf(S).any():
            continue
        elif max(S) < S0 * 2 or min(S) > S0 / 2:
            break
        else:
            pass
    return S, h, q


def generatePath(model, empZ, L, ht1=None, iteration=1000, GarchType="lsngarch"):
    Spath,hs,qs = pd.DataFrame(columns=range(iteration)),pd.DataFrame(columns=range(iteration)),pd.DataFrame(columns=range(iteration))
    for n in range(iteration):
        Spath[n],hs[n],qs[n] = HestonNandiPath(model,empZ,1,L,GarchType,ht1)
    return Spath, hs, qs


def hedgeMCPriceall(TypeFlag, S0, K, model, Spath, hs, qs):
    r_perStep = model.rfr
    Spath = S0 * Spath
    hs2 = np.sqrt(hs)

    nT = len(Spath) # 行数为长度
    rnT = nT-1
    iteration = len(Spath.columns) # 列数才是条数
    cb,costs = np.zeros(iteration),np.zeros(iteration)

    default = {'TypeFlag':TypeFlag,'Strike':K,'Rf':r_perStep}
    option = BSMoption(default)
    for n in range(iteration):
        S,h,q,h2 = Spath[n],hs[n],qs[n],hs2[n]
        Gamma,Delta = np.zeros(rnT),np.zeros(rnT)
        for ii in range(rnT): # nT 总长度 ii 第几期 Nt-ii还剩多少 再减一 ii从0开始的
            t = nT-ii-1
            underlying = S[ii]
            sigma_perStep = h2[ii]
            Gamma[ii] = option.get_gamma(underlying = underlying,sigma = sigma_perStep,t2exp = t)
            Delta[ii] = option.get_delta(underlying = underlying,sigma = sigma_perStep,t2exp = t)
        longOpt = 1
        reb = opt_replicate(S, K, Delta, Gamma, r_perStep, TypeFlag, longOpt)
        X,cost = reb
        cb[n] = X / (1 + r_perStep) ** rnT
        if TypeFlag=="c":
            cb[n] <- min(S[0],max(cb[n],0,S[0]-K/(1+r_perStep)**rnT))
        else:
            cb[n] <- min(K/(1+r_perStep)**rnT,max(cb[n],0,K/(1+r_perStep)**rnT-S[0]))
        costs[n] = cost
    return cb, costs # 我日妈智障啊，检查半天bug，原来这里缩进到循环里了

        # X,cost,cd = 0,0,0 # 初始定价、成本、delta
        # for ii in range(rnT): # rnT比实际长度短1，对冲实际上是从当期开始往回看的，ii为上一期，ii+1为本期
        #     t = nT - ii # 即nT-ii-1
        #     sigma_perStep = h2[ii]
        #     underlying = S[ii]
        #     underlying_now = S[ii+1]
        #     delta = option.get_delta(underlying = underlying,sigma = sigma_perStep,t2exp = t)
        #     gamma = option.get_gamma(underlying = underlying,sigma = sigma_perStep,t2exp = t)

        #     # 进行逐步对冲，替代对冲函数 cd 对冲头寸 cost 成本 X 价格
        #     band = max(1e-3,(3/2 * np.exp(-r_perStep * t) * 5e-4 * underlying * gamma ** 2) ** (1.0/3))
        #     band = min(band,1e-1) * 0 # 乘0为放弃使用band
        #     if np.abs((1-2*longOpt) * delta - cd) > band:
        #         transaction = abs((1 - 2 * longOpt) * delta - cd) * 5e-4
        #         cd = (1 - 2 * longOpt) * delta + band * 0.5 * np.sign(cd - (1 - 2 * longOpt) * delta)
        #         cost += transaction

        #     if X - cd * underlying < 0 :# S[ii+1] 为下一期价格，但是由于
        #         X = cd * underlying_now + (1 + r_perStep) * (X - cd * underlying)
        #     else:# 由于总是设定r_perStep = 0，上下其实没差
        #         X = cd * underlying_now + (X - cd * underlying)
        
        # if TypeFlag == "c":
        #     X = X - (1 - 2 * longOpt) * max(S[rnT] - K, 0)
        # else:
        #     X = X - (1 - 2 * longOpt) * max(K - S[rnT], 0)
        
        # cb[n] = X / (1 + r_perStep) ** rnT
        # if TypeFlag=="c":
        #     cb[n] <- min(S[0],max(cb[n],0,S[0]-K/(1+r_perStep)**rnT))
        # else:
        #     cb[n] <- min(K/(1+r_perStep)**rnT,max(cb[n],0,K/(1+r_perStep)**rnT-S[0]))
        # costs[n] = cost
    # return cb, costs

"""
  S: 股票价格
  K: 行权价格
  r_perStep: 无风险收益率
  TypeFlag: 期权信息, c=Call, p=Put
  longOpt : 方向 0=买权, 1=卖权, 1 - 2*longOpt = 对冲的方向

  cd: 上一次对冲时刻持有的股票的数量, 上一次对冲时的delta
  band: 对冲阈值, 当前 delta 和 cd 有一定差距时进行对冲
  X: 期权复制, 价格等值的(股票 + 现金)组合的价值, cd * S[t] = 当前的股票组合, X - cd * S[t-1] = t-1 时刻的借钱的数量
"""
def opt_replicate(S, K, Delta, Gamma, r_perStep, TypeFlag, longOpt):
    # longopt 0/1
    nT = len(S)
    X = 0
    cost = 0
    cd = 0
    for ii in range(1, nT):
        # t = nT - ii - 1
        # band = max(1e-3, (3 / 2 * np.exp(-r_perStep * t) * 5e-4 * S[ii - 1] * Gamma[ii - 1] ** 2) ** (1.0 / 3))
        # band = min(band, 1e-1)
        band = 0
        if abs((1 - 2 * longOpt) * Delta[ii - 1] - cd) > band:
            transaction = abs((1 - 2 * longOpt) * Delta[ii - 1] - cd) * 5e-4
            cd = (1 - 2 * longOpt) * Delta[ii - 1] + band * 0.5 * np.sign(cd - (1 - 2 * longOpt) * Delta[ii - 1])
            cost += transaction
        X = cd * S[ii] + (1 + r_perStep) * (X - cd * S[ii - 1])

    if TypeFlag == "c":
        X = X - (1 - 2 * longOpt) * max(S[nT - 1] - K, 0)
    else:
        X = X - (1 - 2 * longOpt) * max(K - S[nT - 1], 0)
    return [X, cost]
