from cmath import nan
import numpy as np
import pandas as pd
from scipy.stats import norm


def lsngarch_bayes(x, rfr=0, maxtime = 2000, forgettime = 1000, special = 500):
    index = ['lamda', 'omega', 'alpha', 'beta', 'phi', 'rho', 'gamma1', 'gamma2']
    bsL = pd.Series([-0.0020,0,0.1,0,0,0.9,0,0],index=index)
    bsH = pd.Series([0.0020,0.01,1,0.9,1,1,1,1],index=index)
    sigma = bsH - bsL
    sigma0 = sigma
    params = ls_generatepar(bsL, bsH, rfr, x)
    llh = params['llh']
    params['lamda'] = 0

    
    mat1 = pd.DataFrame(0,index=range(maxtime - forgettime),columns=index+['llh']) # 多加了llh1
    mat1.loc[0,:] = params.values
    for i in range(1, maxtime):# 第一次不做，计数从1开始
        # 每一次都进行参数估计，但前期的没有必要保存
        # if i % 20==0:
        #     print(i)
        params = ls_generatepar_nextstep(bsL, bsH, sigma, params, llh, rfr, x) # lamda不做 永远为0
        if i >= forgettime:
            line = i - forgettime
            mat1.loc[line,:] = params.values
            #最后开始特殊处理,sigma发生变化
            if i >= forgettime+special:
                m2 = mat1.loc[line - 200:line,:].std(axis=0) # 过去特定次数的方差
                sigma[1:] = pd.DataFrame([sigma0[1:]/100,2 * m2[1:-1]]).max(axis=0) # lamda不做 永远为0
    # 取最后一定条数的均值
    param = mat1.iloc[-100:,:8].mean(axis=0)
    model = param.copy()
    model['rfr'] = rfr
    estimate = model.copy()
    # Statistics - Printing:
    sigma2 = model['omega'] / (1 - model['rho'])
    llh, z, h, q = llhaff(params, rfr, x, final=True)
    garch_model = [model,estimate,sigma2,llh,z,h,q,x]
    return garch_model


# 生成初始全部参数
def ls_generatepar(bsL, bsH, rfr, x):
    minq = 0.7*np.var(x)
    maxq = 1.3*np.var(x)
    sat = True
    params = pd.Series(index=['lamda', 'omega', 'alpha', 'beta', 'phi', 'rho', 'gamma1', 'gamma2'],data=0.0)
    while sat:
        for param in params.index:
            params[param] = np.random.uniform(low=bsL[param], high=bsH[param], size=1)
        rho = params['rho']
        omega = params['omega'] = np.random.uniform(low=(1 - rho) * minq, high=(1 - rho) * maxq, size=1)
        q0 = omega / (1 - rho)
        if q0 > maxq or q0 < minq:
            continue
        llh = llhaff(params=params, rfr=rfr, x=x,final=False)
        sat = np.isnan(llh) or np.isinf(llh)
    params['llh'] = llh
    return params


# 每步更新全部参数
def ls_generatepar_nextstep(bsL, bsH, sigma, params, llh, rfr, x):
    param1 = params.copy()
    minq,maxq = 0.7*np.var(x),1.3*np.var(x)
    for param in params.index:
        if param not in ['omega', 'alpha', 'beta', 'phi', 'rho', 'gamma1', 'gamma2']:
            continue
        param1 = params.copy()
        # 配置信息
        mi,ma = bsL[param], bsH[param]
        if param == 'omega':
            rho = param1['rho']
            mi = max(mi, (1 - rho) * minq)
            ma = min(ma, (1 - rho) * maxq)
        elif param == 'rho':
            omega = param1['omega']
            mi = max(mi, (minq - omega) / minq)
            ma = min(ma, (maxq - omega) / maxq)
        sat = True
        while sat:
            param1[param] = produce_param(params[param], sigma[param], mi, ma)
            rho,omega = param1['rho'],param1['omega']
            q0 = omega / (1 - rho)
            if q0 > maxq or q0 < minq:
                continue
            llh1 = llhaff(params=param1, rfr=rfr, x=x, final=False)
            sat = np.isnan(llh1) or np.isinf(llh1)
        # 判断是否接受
        u = np.random.uniform(size=1)
        acceptprob = np.exp(llh1 - llh, dtype=np.float64) if llh1 - llh < 0 else 1
        if u <= acceptprob:
            params,llh = param1,llh1 # 接受单个参数的更新
    params['llh'] = llh
    return params


# 制造一个参数
def produce_param(param, sigma, mL, mH):
    ext = 10
    mL,mH = min(param, mL),max(param, mH)
    sigma = min(sigma, (mH - mL) / ext)
    ximin = ext * (mL - param) / (mH - mL)
    ximax = ext * (mH - param) / (mH - mL)

    while ximin <= 2 and ximax >= -2:
        xi = np.random.normal(size=1)
        if ximin < xi < ximax:
            return param + sigma * xi
    else:
        return np.random.uniform(low=ximin, high=ximax, size=1) * sigma + param


# 整体更新
def llhaff(params, rfr, x, final=False):
    x = np.array(x)
    z,q,h = x.copy(),x.copy(),x.copy()
    lamda,omega,alpha,beta,phi,rho,gamma1,gamma2 = params[:8].values

    h[0] = omega / (1 - rho)
    q[0] = omega / (1 - rho)
    z[0] = (x[0] - rfr - lamda * h[0]) / np.sqrt(h[0])

    error = False
    for i in range(1, len(z)):
        q[i] = omega + rho * q[i - 1] + phi * h[i - 1] * (z[i - 1] ** 2 - 2 * gamma2 * z[i - 1] - 1)
        h[i] = q[i] + beta * (h[i - 1] - q[i - 1]) + alpha * h[i - 1] * (z[i - 1] ** 2 - 2 * gamma1 * z[i - 1] - 1)
        if (h[i] <= 0) or (q[i] <=0):
            error = True
            break
        z[i] = (x[i] - rfr - lamda * h[i]) / np.sqrt(h[i])
    llhaff = sum(np.log(norm.pdf(z) / np.sqrt(h))) if error is False else nan
    if final:
        return [llhaff, z, h, q]
    else:
        return llhaff
