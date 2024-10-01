# Arquivo criado pra fazer a analise das series concatenadas

# -> isso inclui a analise de distribuição de correlação bem como a extração de métricas estatísticas e confecção dos diagramas de taylor-
# -> aqui nao cabe fazer os espectros, acho melhor fazer um outro arquivo pra isso.
import pandas as pd
import xarray as xr
import os
import datetime
import numpy as np
import json
import re
import matplotlib.pyplot as plt

import sys

sys.path.append(
    '../'
)
import model_filt
from read_data import read_exported_series, treat_exported_series, all_series, sep_series
from read_reanalisys import set_reanalisys_dims
from model_data import get_model_region, get_correlation, mapa_corr
import filtro



model_path = '/data3/MOVAR/modelos/REANALISES/'
data_path =  f'/home/bcabral/mestrado/data/'
fig_folder = f'/home/bcabral/mestrado/fig/'

def filt_data(data, sel_hour = 0):
    array_ssh = np.asarray(data['ssh'])
    data_to_filt = np.concatenate((np.full(len(array_ssh)//2, array_ssh.mean()), array_ssh))
    datetime_to_filt = pd.date_range(end=data.index[-1], periods=len(data_to_filt), freq='H')
    data_low = filtro.filtra_dados(data_to_filt, datetime_to_filt, 'low', modelo = False)
    # mudar aqui o valor inicial pra pegar diferentes horas
    # data_low = data_low[9::24]
    data_low = data_low[len(array_ssh)//2:]
    data['low'] = data_low
    daily_data = data[data.index.hour == sel_hour]
    data_low = np.asarray(daily_data['low'])
    data_high = filtro.filtra_dados(data_low, daily_data.index, 'high')

    data_filt = data_high

    return data_filt


def join_series(all_data):
    n = 0
    series = {}
    store = []
    for i in range(len(list(all_data.keys()))-1):
        j = all_data[list(all_data.keys())[i]]
        if list(all_data.keys())[i][:3] != list(all_data.keys())[i+1][:3]:
            store.append(j)
            series[list(all_data.keys())[i]] = pd.concat(store)
            store = []
            # o dicionario retorna aquilo que ele guardou pra esse ponto
        else:
            # armazena o valor do dado pra esse tempo
            store.append(j)
    series['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0'] = all_data['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0']

    return series


def get_all_available_data(path_data = '/home/bcabral/mestrado/data'):
    treated = all_series(path_data)
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


def get_correlation_matrix(lat, lon, data, t0, tf, json_path, years, point_name):
    if os.path.exists(json_path):
        return json_path
    json_dict = {}

    for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
        print(model)

        reanal = {}
        for year in years:
            reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                               , model)
            
        reanalisys = xr.concat(list(reanal.values()), dim="time")

        reanal_subset = get_model_region(lat, lon, model, reanalisys, set_dims = False)
        reanal_subset['ssh'].load()
        reanal_subset = reanal_subset.sel(time=slice(t0, tf))

        if len(reanal_subset['ssh']) > len(data):
            tf2 = tf - datetime.timedelta(days=len(reanal_subset['ssh']) - len(data))
            reanal_subset = reanal_subset.sel(time=slice(t0, tf2))
        elif len(reanal_subset['ssh']) < len(data):
            data = data[:-(len(data) - len(reanal_subset['ssh']))]


        correlation, latlons= get_correlation(data, reanal_subset, fig_folder)

        correlation[np.isnan(correlation)] = -np.inf
        max_index = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        json_dict[model] = latlons[max_index]
        mapa_corr(lon, lat, reanal_subset, correlation, model, point_name, fig_folder)
    
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)


def join_filt_series(series):
    n = 0
    series = {}
    store = []
    for i in range(len(list(all_data.keys()))-1):
        raw_data = all_data[list(all_data.keys())[i]]
        filtered = filt_data(raw_data)
        # se o ultimo horario for antes de 21 h
        date = raw_data.index.normalize().drop_duplicates()
        if len(date) > len(filtered):
            date = date[:-1]
        j = pd.DataFrame(filtered, date, columns=['ssh'])
        if list(all_data.keys())[i][:3] != list(all_data.keys())[i+1][:3]:
            store.append(j)
            series[list(all_data.keys())[i]] = pd.concat(store)
            store = []
            # o dicionario retorna aquilo que ele guardou pra esse ponto
        else:
            # armazena o valor do dado pra esse tempo
            store.append(j)

    # botando o ultimo cara
    store = []
    raw_data = all_data['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0']
    filtered = filt_data(raw_data)
    # se o ultimo horario for antes de 21 h
    date = raw_data.index.normalize().drop_duplicates()
    if len(date) > len(filtered):
        date = date[:-1]
    j = pd.DataFrame(filtered, date, columns=['ssh'])
    store.append(j)
    series['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0'] = pd.concat(store)

    # series['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ'] = all_data['NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0']

    return series


def clean_key(key):
    # Remove números, 'm', espaços e underscores do final da chave
    return re.sub(r'[0-9m_ ]+$', '', key)


all_data = get_all_available_data() # TODO: Precisa mudar aqui de H pra h (deprecated)

series = join_series(all_data)
series = {clean_key(key): value for key, value in series.items()}

# pegar as series
# filtrar as serie
# compor o merged
fseries = join_filt_series(all_data)
fseries = {clean_key(key): value for key, value in fseries.items()}

for point in fseries:
    print(point)
    # tirando pontos dentro de estuario ou com series esquisitas
    if point == 'CANIVETE' or point == 'CAPITANIA DE SALVADOR' or point == 'CIA DOCAS DO PORTO DE SANTANA' \
        or point == 'IGARAPÉ GRD DO CURUÁ' or point == 'PAGA DÍVIDAS I' or point == 'PORTO DE VILA DO CONDE' \
        or point == 'Salvador_glossbrasil' or point == 'SERRARIA RIO MATAPI':
        continue
    json_path = f'/home/bcabral/mestrado/{point}.json'
    data = fseries[point]
    years = range(data.index[0].year, data.index[-1].year +1)
    t0, tf = data.index[0], data.index[-1]
    lat = series[point]['lat'][0]
    lon = series[point]['lon'][0]

    # esse loop abaixo pega a correlacao em area
    # fazendo de novo a parte dos 25 pontos TEM QUE COMENTAR O FOR DOS MODELOS!:
    get_correlation_matrix(lat, lon, data['ssh'], data.index[0], data.index[-1], f'/home/bcabral/mestrado/{point}_25pts.json',
                    years, point)
                    
    for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
        json_dict = {}
        print('pegando ponto de maior correlacao em ' + model)

        reanal = {}
        for year in years:
            reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                               , model)
            
        reanalisys = xr.concat(list(reanal.values()), dim="time")

        reanal_subset = reanalisys.where((reanalisys.latitude < lat +0.5) & 
                              (reanalisys.longitude > lon -0.5) &
                              (reanalisys.latitude > lat -0.5) & 
                              (reanalisys.longitude < lon +0.5) ,
                              drop=True)
        reanal_subset = reanal_subset.sel(time=slice(t0, tf))

        # filtrar o modelo

        filtered_reanal = model_filt.filtra_reanalise(reanal_subset)
        filtered_reanal_normalized = filtered_reanal.assign_coords(time=filtered_reanal['time'].to_index().normalize())


        # dropar o modelo nos momentos em que eu nao tiver dado
        # ind_reanal = filtered_reanal['time'].to_index().normalize()
        ind_reanal = filtered_reanal_normalized['time'].to_index()
        ind_data = data.index

        common_dates = ind_data.intersection(ind_reanal)
        if filtered_reanal_normalized['time'].to_index().duplicated().sum() >0:
            filtered_reanal_normalized = filtered_reanal_normalized.sel(time=~filtered_reanal_normalized['time'].to_index().duplicated())

        filtered_reanal_common = filtered_reanal_normalized.sel(time=common_dates)

        # agora basta fazer a correlacao de cada ponto, plotar, e exportar a latlon do ponto de maior correlacao

        # if len(data) > len(filtered_reanal_common['time']):
        #     data = data[:-1]

        # cortando caso um seja maior que o outro
        if len(filtered_reanal_common.values) > len(data):
            tf2 = tf - datetime.timedelta(days=len(filtered_reanal_common.values) - len(data))
            filtered_reanal_common = filtered_reanal_common.sel(time=slice(t0, tf2))
        elif len(filtered_reanal_common.values) < len(data):
            data = data[:-(len(data) - len(filtered_reanal_common.values))]



        correlation = np.empty((len(filtered_reanal_common.latitude), len(filtered_reanal_common.longitude)))
        latlons = np.zeros((len(filtered_reanal_common.latitude), len(filtered_reanal_common.longitude)), dtype=object)
        for i in range(len(filtered_reanal_common.latitude)):
            for j in range(len(filtered_reanal_common.longitude)):
                model_series = filtered_reanal_common.isel(latitude=i, longitude=j)

                correlation[i, j] = np.corrcoef(data['ssh'], model_series)[0, 1]
                latlons[i,j] = (float(model_series['latitude'].values), float(model_series['longitude'].values))


        correlation[np.isnan(correlation)] = -np.inf
        max_index = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        json_dict[model] = latlons[max_index]
        mapa_corr(lon, lat, reanal_subset, correlation, model +'MEIA_NOITE' , point, fig_folder)
        
        plt.close('all')
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)


    # esse loop abaixo pega as metricas estatisticas considerando o ponto de maxima correlacao



#     frea = []
#     for i in range(len(model_subset.latitude)):
#         for j in range(len(model_subset.longitude)):
#             model_series = model_subset.isel(latitude=i, longitude=j)
#             mod_ssh = model_series['ssh'].values
#             mod_time = model_series['time'].values
#             mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)
#             frea.append(mod_band)

# filtered_reanal_ensemble[:,0,0].values

# for i in range(len(frea)):
#     j = i//10
#     k = i - j*10
#     print((frea[i] != filtered_reanal_ensemble[:,j,k].values).sum())