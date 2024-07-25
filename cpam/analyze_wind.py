import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os
import sys
from scipy import signal
import math


sys.path.append(
    '../'
)
sys.path.append('../old')
from read_data import read_exported_series, treat_exported_series, all_series, sep_series
import filtro
from crosspecs import crospecs

import math

def get_available_data():
    treated = all_series()
    series = {}
    checked_series = {}
    for serie in treated:
        if treated[serie]['lat'][0] > -20:
            continue
        if treated[serie].index[-1] < pd.Timestamp("1980"):
            continue
        if treated[serie].index[0] > pd.Timestamp("2022"):
            continue
        if treated[serie].index[-1] > pd.Timestamp("2022"):
            if len(treated[serie][:'2021-12-31']) < 24*30*6:
                continue
            else:
                treated[serie] = treated[serie][:'2021-12-31']
        series[serie] = treated[serie]
    
    sep_serie = sep_series(series)

    # check depois de ter os nans
    for serie in sep_serie:
        if sep_serie[serie]['lat'][0] > -20:
            continue
        if sep_serie[serie].index[-1] < pd.Timestamp("1980"):
            continue
        if sep_serie[serie].index[0] < pd.Timestamp("1980"):
            if len(sep_serie[serie]['1980-01-01':]) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie]['1980-01-01':]
        if sep_serie[serie].index[0] > pd.Timestamp("2022"):
            continue
        if sep_serie[serie].index[-1] > pd.Timestamp("2022"):
            if len(sep_serie[serie][:'2021-12-31']) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie][:'2021-12-31']

        checked_series[serie] = sep_serie[serie]

    sel_series = {k: v for k, v in checked_series.items() if len(v) >= 24*30*6}

    return sel_series


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilômetros entre dois pontos na superfície da Terra especificados por sua latitude e longitude.

    Parâmetros:
    lat1, lon1: Latitude e longitude do primeiro ponto
    lat2, lon2: Latitude e longitude do segundo ponto

    Retorna:
    Distância entre os dois pontos em quilômetros.
    """
    # Convertendo graus para radianos
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Diferenças das coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula do haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Raio da Terra em km
    R = 6371

    # Distância
    distance = R * c
    return distance


def filt_wind(wind, wind_datetime, xdays = 1/24):
    # from analyze point
    wind_datetime = pd.DatetimeIndex(wind_datetime)
    wind_low = filtro.filtra_dados(wind, wind_datetime, 'low', x_days = xdays)
    # mudar aqui o valor inicial pra pegar diferentes horas
    wind_df = pd.DataFrame({'low':wind_low},index=wind_datetime)
    daily_wind = wind_df[wind_df.index.hour == 12]
    wind_low = np.asarray(daily_wind['low'])
    # if xdays == 1/24:
    #     wind_low = wind_low[12::24]
    # else:
    #     wind_low = wind_low[2::4]
    wind_high = filtro.filtra_dados(wind_low, daily_wind.index, 'high')

    wind_filt = wind_high

    return wind_filt

def filt_data(data):
    # from analyze point
    data_low = filtro.filtra_dados(data['ssh'], data.index, 'low', modelo = False)
    # mudar aqui o valor inicial pra pegar diferentes horas
    # data_low = data_low[9::24]
    data['low'] = data_low
    daily_data = data[data.index.hour==9]
    data_low = np.asarray(daily_data['low'])
    data_high = filtro.filtra_dados(data_low, daily_data.index, 'high')

    data_filt = data_high

    return data_filt


def set_wind_reanalisys_dims(reanalisys, name):
    '''Changes the dimensions of xarray wind reanalisys objects to latitude, longitude, time, and v10

    Args:
        reanalisys (xarray Dataset): Dataset of the reanalisys data
        name (string): Name of the reanalisys

    Returns:
        xarray Dataset: Dataset of the reanalisys data with corrected dimensions names
    '''
    name = name.upper()
    if name == 'ERA-5':
        reanalisys = reanalisys.rename({'latitude': 'lat', 'longitude': 'lon'})
    elif name == 'MERRA2':
        reanalisys = reanalisys.rename({'V10M':'v10'})
    elif name == 'NCEP':
        reanalisys = reanalisys.rename({'vwnd':'v10'})

    return reanalisys


def plot_grid( reanalisys, nome_modelo):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

#     - Latitude: -52,5 a - 42,5
#    - Longitude: -70 a -60-

    ax.set_extent([-57.5, -72.5, -40, -55], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                        color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    lons = reanalisys['lon'].values
    lats = reanalisys['lat'].values
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    # ax.set_title(f'Ponto ')

    # plt.tight_layout()

    plt.title(nome_modelo + "'s grid")
    plt.tight_layout()

    # os.makedirs(f'/Users/breno/mestrado/CPAM/figs/data_mets/', exist_ok=True)
    plt.savefig(f'/Users/breno/mestrado/CPAM/figs/data_mets/points_{nome_modelo}.png')


def convert_lon(lon):
    return np.where(lon < 0, lon + 360, lon)


def convert_longitude_360_to_180(lon):
    return np.where(lon >= 180, lon - 360, lon)


def plot_mask(mask):
# Plotando o contorno preenchido
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))
    contour = ax.contourf(mask['lon'], mask['lat'], mask, cmap='viridis')

    # Adicionar linhas de contorno
    ax.contour(mask['lon'], mask['lat'], mask, colors='k', linewidths=0.5)

    # Adicionar feições cartográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, zorder=0, edgecolor='black')

    # Adicionar barra de cores
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', shrink=0.5, label='Máscara de Terra/Água')

    # Configurar limites do mapa
    ax.set_extent([-70, -60, -52.5, -42.5], crs=ccrs.PlateCarree())

    # Título do mapa
    ax.set_title('Máscara de Terra/Água', fontsize=15)

    plt.show()


models_path = '/Users/breno/mestrado/CPAM/models/'

models = ['ERA-5', 'MERRA2', 'NCEP']
# NEED TO rename every MERRA2 file to end with .nc

## import my reanalisys
# era_path = models_path + '/ERA-5/'

# reanalera = xr.open_mfdataset(era_path+'*.nc')
# reanalmerra = xr.open_mfdataset(models_path + '/MERRA2/'+'*.nc')
reanalncep = xr.open_mfdataset(models_path + '/NCEP/'+'*.nc')

# reanal = reanalera
# model = 'ERA-5'

reanal = reanalncep
model = 'NCEP'
## set its dimensions names

reanal = set_wind_reanalisys_dims(reanal, model)

## Plot the grids of each model

# plot_grid(reanal, model)


# ## select only ocean data

# mask_path = f'{models_path}IMERG_land_sea_mask.nc'
# ds_mask = xr.open_dataset(mask_path)
# ocean_mask = ds_mask['landseamask']

# lat_range = slice(-52.5, -42.5)
# lon_range = slice(-70, -60) # in 0 to 360
# lon_range_mask = slice(convert_lon(-70), convert_lon(-60)) # in 0 to 360

# mask = ocean_mask.sel(lat=lat_range, lon=lon_range_mask)

# lon_original = mask['lon'].values
# lon_convertido = convert_longitude_360_to_180(lon_original)
# mask = mask.assign_coords(lon=lon_convertido)

# mask_interpolated = mask.interp(lat=reanal['v10']['lat'], lon=reanal['v10']['lon'])

# v10_ocean = reanal['v10'].where(mask_interpolated == 100) # maximo de agua = 100

## pega a correlacao do espectro dele com a dos pontos

v10_ocean = reanal['v10']

'''
TODO:

Seria legal ser mais preciso e fixar pegar SEMPRE os horarios das 12z

Pros modelos seria algo do tipo assim:

data_array = np.array(['2017-01-06T18:00:00.000000000', '2017-01-07T00:00:00.000000000',
                       '2017-01-07T06:00:00.000000000', '2018-01-26T12:00:00.000000000',
                       '2018-01-26T18:00:00.000000000', '2018-01-27T00:00:00.000000000'], 
                      dtype='datetime64[ns]')

# Converter para pandas.DatetimeIndex
datetime_index = pd.DatetimeIndex(data_array)

# Filtrar dados cuja hora é 12
filtered_data = datetime_index[datetime_index.hour == 12]
'''

## importando os dados:

# path_data = '/Users/breno/mestrado/resultados/2014/data/'

# data = treat_exported_series(read_exported_series(path_data + 'macae_2014.csv'))
sel_series = get_available_data()

for serie in sel_series:
    if serie == 'TIPLAMm0' or serie == 'Imbituba_2001_20074' or serie == 'Imbituba_2001_20075' \
        or serie == 'Imbituba_2001_200719' or serie =='TERMINAL PORTUÁRIO DA PONTA DO FÉLIXm0' \
        or serie =='ubatuba22' or serie=='ubatuba34' or serie=='TEPORTIm0' or serie=='Ubatuba_gloss22'\
        or serie =='Ubatuba_gloss34' or serie == 'BARRA DE PARANAGUÁ - CANAL DA GALHETAm0'\
        or serie=='PORTO DE PARANAGUÁ - CAIS OESTEm0' or serie=='ilha_fiscal8' or serie=='ilha_fiscal10'\
        or serie == 'ilha_fiscal11': # ja fiz
        continue
    print(f'fazendo {serie}')
    data = sel_series[serie]
    data_point = (data['lat'][0], data['lon'][0])

    data_filt = filt_data(data)
    time_range = slice(data.index[0], data.index[-1])

    v10_ocean = reanal['v10']

    v10_ocean = v10_ocean.sel(time=time_range)


    # aqui eu fiz um data_array com os resultados filtrados
    # tenho que aprender o apply_ufunc, que deve facilitar bastante a vida aqui.

    # corte para a reanalise do ncep
    lat_range = slice(-10, -52.5)
    # lon_range = slice(-70, -55) # in 0 to 360
    lon_range= slice(convert_lon(-70), convert_lon(-30)) # in 0 to 360
    v10_ocean = v10_ocean.sel(lat=lat_range, lon=lon_range)
    lon_original = v10_ocean['lon'].values
    lon_convertido = convert_longitude_360_to_180(lon_original)
    v10_ocean = v10_ocean.assign_coords(lon=lon_convertido)


    lat_vals = v10_ocean['lat'].values
    lon_vals = v10_ocean['lon'].values
    # time_vals = v10_ocean['time'].values[12::24]
    # time_vals = v10_ocean['time'].values[2::4]
    time_vals = np.asarray(pd.DatetimeIndex(v10_ocean['time'])[pd.DatetimeIndex(v10_ocean['time']).hour==12])

    # time_vals = np.asarray(pd.date_range(v10_ocean['time'].values[0], v10_ocean['time'].values[-1], freq='D'))
    # fv10_data = np.empty((len(lat_vals), len(lon_vals), len(time_vals)))
    fv10_data = np.empty((len(lat_vals), len(lon_vals), len(time_vals)))


    for i, lat in enumerate(v10_ocean['lat'].values):
        for j, lon in enumerate(v10_ocean['lon'].values):
            v10a = v10_ocean.sel(lat=lat, lon=lon, method='nearest').values 
            fv10 = filt_wind(v10a, v10_ocean['time'], xdays = 6/24)

            fv10_data[i, j, :] = fv10


    fv10_da = xr.DataArray(
        fv10_data,
        #coords=[lat_vals, lon_vals, time_vals],
        coords=[lat_vals, lon_vals, time_vals],
        dims=["lat", "lon", "time"],
        name="fv10"
    )


    ''' 
    duas formas de comparar:

    1 - soma o coeficiente de correlacao cruzada de todo o espectro:
    > coef.sum()

    2 - soma o coeficiente de correlacao cruzada somente onde ha confianca esatistica:
    > coef[coef>conf].sum()
    '''


    pta_max = 0
    abs_max = 0

    pts_max = 0
    sup_max = 0

    abss = []
    sups = []

    # partindo pra correlacao cruzada
    n_lat = len(fv10_da['lat'].values)
    n_lon = len(fv10_da['lon'].values)
    abss_data = np.empty((n_lat, n_lon))
    sups_data = np.empty((n_lat, n_lon))
    # Iterar sobre todas as latitudes e longitudes para calcular a correlação cruzada


    for i, lat in enumerate(fv10_da['lat'].values):
        for j, lon in enumerate(fv10_da['lon'].values):
            if len(data_filt) > len(fv10_da.sel(lat=lat, lon=lon).values):
                data_filt = data_filt[:-1]

            serie1 = data_filt
            serie2 = fv10_da.sel(lat=lat, lon=lon).values


            xx1 = serie1
            xx2 = serie2
            ppp = len(xx1)
            dt = 24  # diário
            win = 2
            smo = 999
            ci = 99
            h1, h2, fff, coef, conf, fase = crospecs(xx1, xx2, ppp, dt, win, smo, ci)

            abss_value = coef.sum()
            abss_data[i, j] = abss_value

            sup_value = coef[coef > conf].sum()
            sups_data[i, j] = sup_value

            if abss_value > abs_max:
                abs_max = abss_value
                pta_max = (lat, lon)
            if sup_value > sup_max:
                sup_max = sup_value
                pts_max = (lat, lon)


    abss_da = xr.DataArray(
        abss_data,
        coords=[lat_vals, lon_vals],
        dims=["lat", "lon"],
        name="abss"
    )

    sups_da = xr.DataArray(
        sups_data,
        coords=[lat_vals, lon_vals],
        dims=["lat", "lon"],
        name="sups"
    )
    # Plotar o colormap de abss

    # Plotar o colormap de abss em um mapa
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    abss_da.plot(ax=ax, cmap="viridis", transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')

    ax.plot(data_point[1], data_point[0], 'ro', markersize=10,
            transform=ccrs.PlateCarree(), label=f'Ponto de comparação')

    ax.legend(loc='lower right')

    gl = ax.gridlines(draw_labels = True, color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("Colormap de abss")

    # plt.show()
    plt.savefig('/Users/breno/mestrado/CPAM/figs/corr_maps/ncep/abs/' + serie + '_abs.png')


    # Plotar o colormap de sups em um mapa
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    sups_da.plot(ax=ax, cmap="viridis", transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS, edgecolor='black')

    ax.plot(data_point[1], data_point[0], 'ro', markersize=10,
            transform=ccrs.PlateCarree(), label=f'Ponto de comparação')

    ax.legend(loc='lower right')

    gl = ax.gridlines(draw_labels = True, color='gray', linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("Colormap de sups")

    # plt.show()
    plt.savefig('/Users/breno/mestrado/CPAM/figs/corr_maps/ncep/sup/' + serie + '_sup.png')


# plota o espectro em determinado ponto

# lat = pts_max[0]
# lon = pts_max[1]

# v10a = v10_ocean.sel(lat=lat, lon=lon, method='nearest').values # testando pra um ponto
# fv10 = filt_wind(v10a, data)

    lat = -40
    lon = -57

    # lat = pts_max[0]
    # lon = pts_max[1]
    serie1 = data_filt
    serie2 = fv10_da.sel(lat=lat, lon=lon, method = 'nearest').values

    xx1=serie1
    xx2=serie2
    ppp=len(xx1)
    dt=24#diario
    win=2
    smo=999
    ci=99
    h1,h2,fff,coef,conf,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)
    fig = plt.figure(figsize=(8,12))
    plt.plot(1./fff/24,coef,'b')
    plt.plot(1./fff/24,conf,'--k')
    plt.xlim([0,40])
    plt.grid()
    plt.savefig('/Users/breno/mestrado/CPAM/figs/cross_corr/ped/' + serie + '.png')


    lat = pts_max[0]
    lon = pts_max[1]
    serie1 = data_filt
    serie2 = fv10_da.sel(lat=lat, lon=lon, method = 'nearest').values

    xx1=serie1
    xx2=serie2
    ppp=len(xx1)
    dt=24#diario
    win=2
    smo=999
    ci=99
    h1,h2,fff,coef,conf,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)
    fig = plt.figure(figsize=(8,12))
    plt.plot(1./fff/24,coef,'b')
    plt.plot(1./fff/24,conf,'--k')
    plt.xlim([0,40])
    plt.grid()
    plt.savefig('/Users/breno/mestrado/CPAM/figs/cross_corr/max/' + serie + '.png')

    lat = pta_max[0]
    lon = pta_max[1]
    serie1 = data_filt
    serie2 = fv10_da.sel(lat=lat, lon=lon, method = 'nearest').values

    xx1=serie1
    xx2=serie2
    ppp=len(xx1)
    dt=24#diario
    win=2
    smo=999
    ci=99
    h1,h2,fff,coef,conf,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)
    fig = plt.figure(figsize=(8,12))
    plt.plot(1./fff/24,coef,'b')
    plt.plot(1./fff/24,conf,'--k')
    plt.xlim([0,40])
    plt.grid()
    plt.savefig('/Users/breno/mestrado/CPAM/figs/cross_corr/sup/' + serie + '.png')
    
    plt.close('all')



# pega a correlacao com o lag pra ver a quantidade de dias de atraso do sinal

x = serie1
y = serie2
correlation = signal.correlate(x, y , mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
lag = lags[np.argmax(abs(correlation))]

# verificar a distancia:



tempo = lag * 24 * 60 * 60 # dias para segundos
velocidade = 10

distancia = velocidade * tempo/1000

distancia_reta= haversine(data_point[0], data_point[1], lat, lon)

# seria muito bacana achar um lugar onde eu pudesse calcular distancia entre duas lat lons

# a minha tentativa da correlacao nao foi muito boa, nao sei muito bem como fazer pra selecionar o melhor ponto
# agora!

# talvez seja legal pegar os valores do crospecs (coef) e somar pra ver qual ponto teria o maior.


np.corrcoef(x[:-lag], y[lag:])




# TESTE PRA VER A COBERTURA DE TODOS OS DADOS


def get_all_available_data():
    treated = all_series()
    series = {}
    checked_series = {}
    for serie in treated:
        # if treated[serie]['lat'][0] > -20:
        #     continue
        if treated[serie].index[-1] < pd.Timestamp("1995"):
            continue
        if treated[serie].index[0] > pd.Timestamp("2020"):
            continue
        if treated[serie].index[-1] > pd.Timestamp("2020"):
            if len(treated[serie][:'2019-12-31']) < 24*30*6:
                continue
            else:
                treated[serie] = treated[serie][:'2019-12-31']
        series[serie] = treated[serie]
    
    sep_serie = sep_series(series)

    # check depois de ter os nans
    for serie in sep_serie:
        # if sep_serie[serie]['lat'][0] > -20:
        #     continue
        if sep_serie[serie].index[-1] < pd.Timestamp("1995"):
            continue
        if sep_serie[serie].index[0] < pd.Timestamp("1995"):
            if len(sep_serie[serie]['1995-01-01':]) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie]['1995-01-01':]
        if sep_serie[serie].index[0] > pd.Timestamp("2020"):
            continue
        if sep_serie[serie].index[-1] > pd.Timestamp("2020"):
            if len(sep_serie[serie][:'2019-12-31']) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie][:'2019-12-31']

        checked_series[serie] = sep_serie[serie]

    sel_series = {k: v for k, v in checked_series.items() if len(v) >= 24*30*6}

    return sel_series


sel_infos = {}

for serie in sel_series:
    sel_infos[serie] = (sel_series[serie].index[0], sel_series[serie].index[-1], sel_series[serie]['lat'][0])
    # print(serie, all_data[serie].index[0],all_data[serie].index[-1] )
data_info = pd.DataFrame(sel_infos).T
data_info.columns = ['Data Inicial', 'Data Final', 'Latitude']
data_info = data_info.sort_values('Latitude')
data_info.to_csv('/Users/breno/mestrado/CPAM/data_info.csv')


all_data = get_all_available_data()


all_infos = {}

for serie in all_data:
    all_infos[serie] = (all_data[serie].index[0], all_data[serie].index[-1], all_data[serie]['lat'][0])
    # print(serie, all_data[serie].index[0],all_data[serie].index[-1] )
data_info = pd.DataFrame(all_infos).T
data_info.columns = ['Data Inicial', 'Data Final', 'Latitude']
data_info = data_info.sort_values('Latitude')
data_info.to_csv('/Users/breno/mestrado/resultados/data_info.csv')