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

import numpy as np

def stats(model, data):
    '''Função que faz as estatísticas de comparação entre dados e modelo.

    Args:
        model (numpy.ndarray): array dos outputs do modelo filtrados
        data (numpy.ndarray): array dos dados filtrados

    Returns:
        tuple: Resultados das estatísticas de comparação entre dados e modelo
    '''
    if (data/model).mean() > 50 or (data/model).mean() < -50:
        print('Checar se os dados e o modelo estao na mesma unidade')

    sum_data = np.sum(data)
    sum_model = np.sum(model)

    mean_data = np.mean(data)
    mean_model = np.mean(model)

    corr =round(np.corrcoef(model, data)[0,1] * 100, 2)

    # rmse = np.sqrt(np.sum((model - data)**2) /len(data))
    
    nrmse  =  np.sqrt((((sum_model - sum_data)**2)/(sum_data**2)))

    si_up = ((sum_model - mean_model) - (sum_data - mean_data))**2
    si_down = sum_data**2
    si = si_up/si_down

    # bias = np.mean(data - model) # ver a forma de fazer o nbias

    nbias = (sum_model - sum_data) / sum_data

    return(corr, nbias, nrmse, si)


def dependent_stats(model, data):
    '''Faz as estatísticas dependentes de comparação entre dados e modelo.

    Args:
        model (numpy.ndarray): array dos outputs do modelo filtrados
        data (numpy.ndarray): array dos dados filtrados

    Returns:
        tuple: Resultados das estatísticas de comparação entre dados e modelo
    '''
    if (data/model).mean() > 50 or (data/model).mean() < -50:
        print('Checar se os dados e o modelo estao na mesma unidade')

    sum_data = np.sum(data)
    sum_model = np.sum(model)

    mean_data = np.mean(data)
    mean_model = np.mean(model)

    n = len(data)

    bias = (sum_model - sum_data) / n
    
    rmse  =  np.sqrt((((sum_model - sum_data)**2)/n))

    scrmse = np.sqrt(rmse**2 - bias**2)

    # bias = np.mean(data - model) # ver a forma de fazer o nbias


    return(bias, rmse, scrmse)


def general_stats(series):
    '''Faz as estatísticas gerais de uma série.

    Args:
        series (numpy.ndarray): array da série
    '''

    max = np.max(series)

    min = np.min(series)

    sum = np.sum(series)

    mean = np.mean(series)

    std = np.std(series)

    median = np.median(series)

    return(sum, mean, std, median, max, min)


def filt_data(data, sel_hour = 21):
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

def get_data_stats(series, place):
    gen_stats_point = general_stats(series)
    gen_stat_df = pd.DataFrame(gen_stats_point, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)
    # gen_stat_df = gen_stat_df.rename(columns={0: place})
    os.makedirs(f'{fig_folder}/n_stats/{place}/', exist_ok=True)
    gen_stat_df.to_csv(f'{fig_folder}/n_stats/{place}/gen_{place}.csv')


def get_reanalisys_stats(data, reanalisys, place, model):
    stat_met = stats(reanalisys, data)
    dep_stat_met = dependent_stats(reanalisys, data)
    gen_stats = general_stats(reanalisys)

    # gen_stats['Dado'] = general_stats(data)

    stat_df = pd.DataFrame(stat_met, index=['CORR', 'NBIAS', 'NRMSE', 	'SI']).round(2)
    dep_stat_df = pd.DataFrame(dep_stat_met, index=['BIAS', 	'RMSE','SCRMSE']).round(2)
    gen_stat_df = pd.DataFrame(gen_stats, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)

    # TODO: acho que seria mais legal juntar todas essas metricas em um so arquivo. Mais facil mexer depois
    os.makedirs(f'{fig_folder}/n_stats/{place}/', exist_ok=True)
    stat_df.to_csv(f'{fig_folder}/n_stats/{place}/comp_{model}.csv')
    dep_stat_df.to_csv(f'{fig_folder}/n_stats/{place}/dep_{model}.csv')
    gen_stat_df.to_csv(f'{fig_folder}/n_stats/{place}/gen_{model}.csv')



def get_model_region(lat, lon, model_name, reanalisys, set_dims = True):
    # reanalisys = xr.open_mfdataset(model_path + '/*SSH*.nc')
    if set_dims:
        reanalisys = set_reanalisys_dims(reanalisys, model_name)

    lat_idx = np.abs(reanalisys['latitude'] - lat).argmin().item()
    lon_idx = np.abs(reanalisys['longitude'] - lon).argmin().item()

    lat_indices = range(lat_idx-3, lat_idx+2)
    lon_indices = range(lon_idx-2, lon_idx+3)
    reanal_subset = reanalisys.isel(latitude=lat_indices, longitude=lon_indices)

    return reanal_subset


def get_correlation(filt_data, model_subset, fig_folder):
    correlation = np.empty((len(model_subset.latitude), len(model_subset.longitude)))
    latlons = np.zeros((len(model_subset.latitude), len(model_subset.longitude)), dtype=object)
    for i in range(len(model_subset.latitude)):
        for j in range(len(model_subset.longitude)):
            model_series = model_subset.isel(latitude=i, longitude=j)
            mod_ssh = model_series['ssh'].values
            mod_time = model_series['time'].values
            mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)
            correlation[i, j] = np.corrcoef(filt_data, mod_band)[0, 1]
            latlons[i,j] = (float(model_series['latitude'].values), float(model_series['longitude'].values))
    return correlation, latlons


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

    data = fseries[point]
    years = range(data.index[0].year, data.index[-1].year +1)
    t0, tf = data.index[0], data.index[-1]
    lat = series[point]['lat'][0]
    lon = series[point]['lon'][0]

    # esse loop abaixo o ponto indicado pelo json como maior correlacao para a reanalise
    get_data_stats(data['ssh'], point)

    for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
        try:
            print('pegando estatisticas para ' + model)
            point_info = json.load(open(f'/home/bcabral/mestrado/{point}_25pts.json'))
            reanal = {}
            for year in years:
                reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                                    , model)
                
            reanalisys = xr.concat(list(reanal.values()), dim="time")
            latlon = point_info[model]
            model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
            model_series = model_series.sel(time=slice(t0, tf))

            filtered_reanal = model_filt.filtra_reanalise(model_series)

            filtered_reanal_normalized = filtered_reanal.assign_coords(time=filtered_reanal['time'].to_index().normalize())


            # dropar o modelo nos momentos em que eu nao tiver dado
            ind_reanal = filtered_reanal_normalized['time'].to_index()
            ind_data = data.index

            common_dates = ind_data.intersection(ind_reanal)
            if filtered_reanal_normalized['time'].to_index().duplicated().sum() >0:
                filtered_reanal_normalized = filtered_reanal_normalized.sel(time=~filtered_reanal_normalized['time'].to_index().duplicated())

            filtered_reanal_common = filtered_reanal_normalized.sel(time=common_dates)


            # cortando caso um seja maior que o outro
            if len(filtered_reanal_common.values) > len(data):
                tf2 = tf - datetime.timedelta(days=len(filtered_reanal_common.values) - len(data))
                filtered_reanal_common = filtered_reanal_common.sel(time=slice(t0, tf2))
            elif len(filtered_reanal_common.values) < len(data):
                data = data[:-(len(data) - len(filtered_reanal_common.values))]


            # agora basta extrair as metricas estatisticas:
            get_reanalisys_stats(data['ssh'], filtered_reanal_common.values * 100, point, model)
        except Exception as e:
            print('Nao conseguiu fazer :(')
            print(e)
            print('continuando!')
