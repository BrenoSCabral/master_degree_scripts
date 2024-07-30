# common libs
import os
import xarray as xr
import numpy as np
import json
import sys
import pandas as pd

# my files
from read_data import read_exported_series, treat_exported_series, all_series, sep_series
from model_data import get_model_region, get_correlation
from read_reanalisys import set_reanalisys_dims
import filtro
sys.path.append(
    'old'
)
import stats
import general_plots as gplots


# setar variaveis
# talvez seja melhor botar isso na chamada da funcao, tipo aquele argv que o jonas fez
# year = 2012
server = True

if server:
    model_path = '/data3/MOVAR/modelos/REANALISES/'
    data_path =  f'/home/bcabral/mestrado/data/'
    fig_folder = f'/home/bcabral/mestrado/fig/'
    # data_path = f'/home/bcabral/mestrado/data/{year}/'
    # fig_folder = f'/home/bcabral/mestrado/fig/{year}'

# else:
#     data_path  = f'/Users/breno/mestrado/resultados/{year}/data/'
#     fig_folder  = f'/Users/breno/mestrado/resultados/{year}/figs/'
#     # fig_folder = f'/Users/breno/Documents/Mestrado/resultados/{year}/figs'
    


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


def get_data(point):
    data = read_exported_series(point)

    lat = data['lat'][0]
    lon = data['lon'][0]



    return data, lat, lon


def contar_nans_seguidos(serie):
    is_nan = serie.isna()
    shift_is_nan = is_nan.shift(fill_value=False)
    grupos = (is_nan != shift_is_nan).cumsum()
    contagem_grupos = is_nan.groupby(grupos).sum()
    contagem_grupos = contagem_grupos[contagem_grupos > 0]
    return contagem_grupos


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

    return data_filt


def get_data_stats(series, place):
    gen_stats_point = stats.general_stats(series)
    gen_stat_df = pd.DataFrame(gen_stats_point, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)
    os.makedirs(f'{fig_folder}/stats/{place}/', exist_ok=True)
    gen_stat_df.to_csv(f'{fig_folder}/stats/{place}/gen_{place}.csv')
    

def get_correlation_matrix(lat, lon, data, t0, tf, json_path, years):
    if os.path.exists(json_path):
        return json_path
    json_dict = {}

    for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
        print(model)
        # setando caminho do servidor
        '''
        TESTAR ISSO AQUI
        ->-
        ->-
        ->-
        '''
        # TODO: TEM QUE TESTAR
        # reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
        reanal = {}
        for year in years:
            reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                               , model)
            
        reanalisys = xr.concat(list(reanal.values()), dim="time")

        reanal_subset = get_model_region(lat, lon, model, reanalisys)
        reanal_subset['ssh'].load()
        reanal_subset = reanal_subset.sel(time=slice(t0, tf))


        correlation, latlons= get_correlation(data, reanal_subset, fig_folder)

        correlation[np.isnan(correlation)] = -np.inf
        max_index = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        json_dict[model] = latlons[max_index]
        # mapa_corr(lon, lat, reanal_subset, correlation, model, '_'.join(str.split(data_name, '_')[:-1]), fig_folder)
    
    with open(json_path, 'w') as f:
        json.dump(json_dict, f)


def get_reanalisys(point, model, di, df, years):
    point_info = json.load(open(f'/home/bcabral/mestrado/{point}.json'))
    reanal = {}
    for year in years:
        reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                            , model)
        
    reanalisys = xr.concat(list(reanal.values()), dim="time")
    latlon = point_info[model]
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
    model_series = model_series.sel(time=slice(di, df))

    mod_ssh = model_series['ssh'].values
    # nao tava dando problema entao n ha necessidade de fazer assim
    # mod_ssh_to_filt = np.concatenate((np.full(len(mod_ssh)//2, mod_ssh.mean()), mod_ssh))
    # mod_time = pd.date_range(end=model_series['time'].values[-1], periods=len(mod_ssh_to_filt), freq='D')
    mod_time = model_series['time'].values
    mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)

    return mod_ssh, mod_band, mod_time


def get_reanalisys_stats(data, reanalisys, place, model):
    stat_met = stats.stats(reanalisys * 100, data)
    dep_stat_met = stats.dependent_stats(reanalisys * 100, 		data)
    gen_stats = stats.general_stats(reanalisys * 100)

    # gen_stats['Dado'] = stats.general_stats(data)

    stat_df = pd.DataFrame(stat_met, index=['CORR', 'NBIAS', 'NRMSE', 	'SI']).round(2)
    dep_stat_df = pd.DataFrame(dep_stat_met, index=['BIAS', 	'RMSE','SCRMSE']).round(2)
    gen_stat_df = pd.DataFrame(gen_stats, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)

    # TODO: acho que seria mais legal juntar todas essas metricas em um so arquivo. Mais facil mexer depois
    os.makedirs(f'{fig_folder}/stats/{place}/', exist_ok=True)
    stat_df.to_csv(f'{fig_folder}/stats/{place}/comp_{model}.csv')
    dep_stat_df.to_csv(f'{fig_folder}/stats/{place}/dep_{model}.csv')
    gen_stat_df.to_csv(f'{fig_folder}/stats/{place}/gen_{model}.csv')


def main():
    # se os warning tiverem enchendo mto o saco
    import warnings
    warnings.filterwarnings('ignore')
    
    all_data = get_all_available_data() # TODO: Precisa mudar aqui de H pra h (deprecated)

    for point in all_data:
        data = all_data[point]
        if data['ssh'].max() < 1: # passando a unidade que creio ser m para cm
            data['ssh'] = data['ssh'] * 100
        lat, lon = data['lat'][0], data['lon'][0]
        data_filt = filt_data(data)

    # for point in os.listdir(data_path):
    #     if point[-1] != 'v':
    #         continue

    #     data, lat, lon = get_data(data_path + point)
    #     if len(data) > 10000:
    #         data = treat_exported_series(read_exported_series(data_path + point))

    #     max_nans = contar_nans_seguidos(data['ssh']).max()
    #     if max_nans > 6:
    #         continue
    #     elif max_nans > 0:
    #         data['ssh'] = data['ssh'].interpolate('quadratic')

    #     if data['ssh'].max() < 2.0:
    #         # mudar essa funcao aqui. Pegar o valor maximo da serie e se ele for menor que n,
    #         # multiplicar por 100
    #         data['ssh'] = data['ssh'] *100

    #     data_filt = filt_data(data)

    #     print('OK ' + point)


        gplots.plot_time_series(data['ssh'], 'Série Temporal de ' + point, f'{fig_folder}/ponto_serie/', point)
        # nomear eixos (cm e data)
        gplots.plot_spectrum(data['ssh'], f'Espectro de {point}', f'{fig_folder}/spectra/{point}/', f'spec_{point}')
        # tirar label
        gplots.plot_double_spectrum(data['ssh'], data_filt, f'Medido vs Filtrado ({point})', f'{fig_folder}/spectra/{point}/', f'comp_spec_{point}')

        get_correlation_matrix(lat, lon, data_filt, data.index[0], data.index[-1], f'/home/bcabral/mestrado/{point}.json',
                               range(data.index[0].year, data.index[-1].year +1))
        get_data_stats(data_filt, point)
        for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
            data_filt_m = data_filt
            print(model)
            reanalisys, fil_reanalisys, filt_time = get_reanalisys(point, model, data.index[0],
                                                                   data.index[-1], range(data.index[0].year, data.index[-1].year +1))
            if pd.to_datetime(reanalisys['time'].values[0]).hour != 0:
                data_filt_m = filt_data(data, hour = pd.to_datetime(reanalisys['time'].values[0]).hour -3)

            
            reanalisys = reanalisys * 100 # pasando pra cm
            fil_reanalisys = fil_reanalisys * 100 # passando pra cm

            gplots.plot_spectrum(reanalisys, f'Espectro de {point} ({model})', f'{fig_folder}/spectra/{point}/', f'spec_{model}', is_data = False)
            # ajustar a unidade acima
            gplots.plot_double_spectrum(reanalisys, fil_reanalisys, f'Original vs Filtrado ({point} - {model})',
                                        f'{fig_folder}/spectra/{point}/',f'comp_spec_{model}', is_data = False)
            
            # gplots.compare_spectra(data, reanalisys, f'Espectro {point} vs {model}', f'{fig_folder}/spectra/{point}',f'crosspec_{model}')
            gplots.compare_spectra(data_filt_m, fil_reanalisys, f'Espectro Filtrado {point} vs {model}', f'{fig_folder}/spectra/{point}/',f'crosspec_filt_{model}')
            gplots.plot_double_spectrum(data_filt_m, fil_reanalisys, f'Filtrado Ponto vs Modelo ({point} - {model})',
                                        f'{fig_folder}/spectra/{point}/',f'comp_{point}_{model}', is_data = False)
            
            
            gplots.compare_time_series(data_filt_m, fil_reanalisys, f'{point} vs {model}', f'{fig_folder}/series/{point}/',f'{model}')
            # faltam labels e botar na mesma unidade o modelo e o ponto

            get_reanalisys_stats(data_filt_m, fil_reanalisys, point, model)
