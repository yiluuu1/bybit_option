from lsngarch_bayes import *
from hedgePrice import *
from PlainVanillaOptions import *
import numpy as np
from scipy import interpolate

def train(index_price, futures_price, strike, t2exp, option_type, lens, GarchType):
    ivs = np.array([])

    R = np.log(index_price.pct_change() + 1)[1:]
    R.index = np.arange(start=0, stop=len(R))

    # training models
    m = lsngarch_bayes(x=R)
    ht1 = [m.h[len(m.h) - 287:].mean(), m.q[len(m.q) - 287:].mean()]

    # simulate path
    maxl = max(t2exp*1440/5)  #t2exp are calculate in days


    spath = generatePath(model=m.model, empZ=m.z, L=maxl, ht1=ht1, GarchType=GarchType)
    for i in range(lens):
        s = futures_price[i]
        k = strike[i]
        L = t2exp / 5
        if (s > k and option_type[i] == 'c') or (s < k and option_type[i] == 'p'):
            continue
        cbs = hedgeMCPriceall(s, k, m.model, L, spath, option_type[i], ht1=ht1, GarchType=GarchType)
        option_price = np.mean(cbs)
        option_price_std = np.std(cbs)
        option_price_median = np.quantile(cbs, 0.5)
        # if garchtype== "agarchml":
        #     tp =HNGOption(row['type'], model, s, k, L, 0, ht1=ht1)
        #     row["mltp"] =tp
        iv = GBSVolatility(option_price, option_type[i], s, k,t2exp/ 365, 0, 0)
        ivs=np.append(ivs, iv)

    #fit the curve
    moneyness=np.log(futures_price/strike)/np.sqrt(t2exp/365)
    s = interpolate.CubicSpline(moneyness, ivs)
    return s

def fetch_iv(s, futures_price, strike, t2exp):
    moneyness = np.log(futures_price / strike) / np.sqrt(t2exp / 365)
    iv=s(moneyness)
    return iv

if __name__ == '__main__':
    index_price, futures_price, strike, t2exp, option_type, lens=pd.read_csv('some how')
    GarchType='lsngarch'
    s=train(index_price, futures_price, strike, t2exp, option_type, lens, GarchType)

    f, k, t=pd.read_csv('some how')
    res=fetch_iv(s, f, k, t)