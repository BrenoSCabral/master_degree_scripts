import pandas as pd


path_g = '/Users/breno/Documents/Mestrado/dados/gloos/goos/'

def trata_serie(serie):
    # serie.loc[len(serie)] = serie.columns.to_list()
    serie.columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg', 'ssh']

    d_index = []
    for i in range(len(serie)):
        ano = serie['yyyy'][i]
        mes = serie[' mm'][i]
        dia = serie[' dd'][i]
        hora = serie [' hour'][i]
        minu = serie[' min'][i]
        seg = serie[' seg'][i]
        date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:{seg}')
        d_index.append(date)

    serie['data'] = d_index
    serie.set_index('data', inplace=True)
    serie.drop(columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg'], inplace=True)


    serie = serie.sort_index()
    # serie['lat'] = ponto[0]
    # serie['lon'] = ponto[1]


    serie['ssh'] = serie['ssh'].astype(float)/10
    serie = serie.mask(serie < -1000)
    return(serie)


ilf = trata_serie(pd.read_csv('/Volumes/BRENO_HD/dados_mestrado/dados/' + 'ilha_fiscal.csv'))

uba = trata_serie(pd.read_csv(path_g + 'ubatuba.csv'))

imb = trata_serie(pd.read_csv(path_g + 'imbituba.csv'))

rs = trata_serie(pd.read_csv(path_g + 'rio_grande.csv'))


import xarray as xr



# to_compare:

ilf14 = ilf[ilf.index.year == 2014]
uba14 = uba[uba.index.year == 2014]
imb14 = imb[imb.index.year == 2014]

# reamostrado
ilf14r = ilf14[ilf14.index.hour == 0]
ilf14r = ilf14r[ilf14r.index.minute == 0]
ilf14r = ilf14r[ilf14r.index.second == 0]


uba14r = uba14[uba14.index.hour == 0]
uba14r = uba14r[uba14r.index.minute == 0]
uba14r = uba14r[uba14r.index.second == 0]

imb14r = imb14.resample('D').mean()
imb14r = imb14[imb14.index.hour == 0]
imb14r = imb14r[imb14r.index.minute == 0]
imb14r = imb14r[imb14r.index.second == 0]

import numpy as np
from matplotlib import pyplot as plt

def analise(serie, modelo):
    xx1= np.asarray(serie)
    xx2= modelo
    ppp=len(xx1)
    dt=24 # diario
    win=2
    smo=999
    ci=99
    h1,h2,fffg,coefg,confg,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)



    plt.semilogx(1./fffg/24,coefg,'b')
    plt.semilogx(1./fffg/24,confg,'--k')
    plt.grid()
    plt.xlim([2,40])
    plt.show()


mask = np.isnan(imb14r['ssh'].values)
imb14r['ssh'].values[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), imb14r['ssh'].values[~mask])


ncar = xr.open_mfdataset('/Users/breno/Downloads/vwnd.10m.gauss.2014.nc')
ncar = ncar.rename({'vwnd':'v'})

ncarv = ncar['v'].sel(lat=-45, lon=-57, method='nearest')
ncarv0= ncarv.values[::4]
analise(imb14r['ssh'], ncarv0)

# analise(uba14r['ssh'], ncarv0)
# analise(imb14r['ssh'], ncarv0)

era5 = xr.open_mfdataset('/Users/breno/Downloads/era5.nc')
era5 = era5.rename({'v10':'v'})

era5v = era5['v'].sel(latitude=-45, longitude=-57, method='nearest')
era5v0= era5v.values[::24]


analise(ilf14r['ssh'], era5v0)

# from matplotlib import pyplot as plt