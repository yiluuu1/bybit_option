from lsngarch_bayes import *
from hedgePrice import *
from HestonNandiOptions_MY import *
from PlainVanillaOptions import *
import numpy as np
from datetime import *
import pandas as pd
from scipy import interpolate

futures_file = 'test_futures.csv'
options_file = ['test_option.csv']


def optprices(optinfo, model, ht1, garchtype):
    maxl = 0
    for index, row in optinfo.iterrows():
        t1 = datetime.strptime(row["DateTime"], '%Y/%m/%d %H:%M')
        t2 = datetime.strptime(row["ExpDateTime"], '%Y/%m/%d %H:%M')
        L = round((t2 - t1).total_seconds() / 60 / 5)
        maxl = max(maxl, L)

    spath = generatePath(model=model.model, empZ=model.z, L=maxl, ht1=ht1, GarchType=garchtype)
    for index, row in optinfo.iterrows():
        s = row['Fprice']
        k = row['Strike']
        t1 = datetime.strptime(row["DateTime"], '%Y/%m/%d %H:%M')
        t2 = datetime.strptime(row["ExpDateTime"], '%Y/%m/%d %H:%M')
        L = round((t2 - t1).total_seconds() / 60 / 5)
        if (s > k and row['type'] == 'c') or (s < k and row['type'] == 'p'):
            continue
        cbs = hedgeMCPriceall(s, k, model.model, L, spath, row['type'], ht1=ht1, GarchType=garchtype)
        row["mean"] = np.mean(cbs)
        row["std"] = np.std(cbs)
        row["median"] = np.quantile(cbs, 0.5)
        if garchtype== "agarchml":
            tp =HNGOption(row['type'], model, s, k, L, 0, ht1=ht1)
            row["mltp"] =tp
    return optinfo

def iv_compute(futures_file, options_file):
    futures_data = pd.read_csv(futures_file)['mean']
    options_data = pd.read_csv(options_file)

    R = np.log(futures_data.pct_change() + 1)[1:]
    R.index = np.arange(start=0, stop=len(R))

    m = lsngarch_bayes(x=R)

    ht1 = [m.h[len(m.h) -100:].mean(), m.q[len(m.q) -100:].mean()]
    optinfo = optprices(options_data, m, ht1, 'lsngarch')

    return optinfo


if __name__ == '__main__':

    for option in options_file:
        data = iv_compute(futures_file, option)

        ivs=np.array([])
        for index, row in data.iterrows():
            iv = GBSVolatility(row['mean'], row['type'], row['Fprice'], row['Strike'],row['T2exp']/ 365, 0, 0)
        x = data['Moneyness']
        s = interpolate.CubicSpline(x, ivs)
