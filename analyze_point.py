# common libs
import os
import xarray as xr
import numpy as np
import json
import sys
import pandas as pd

# my files
from read_data import read_exported_series
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
year = 2009
server = False

if server:
    model_path = '/data3/MOVAR/modelos/REANALISES/'
    data_path = f'/home/bcabral/mestrado/data/{year}/'
    fig_folder = f'/home/bcabral/mestrado/fig/{year}'
else:
    data_path  = f'/Users/breno/Documents/Mestrado/resultados/{year}/data/'
    fig_folder = f'/Users/breno/Documents/Mestrado/resultados/{year}/figs'


# Tem que testar toda essa brincadeira aqui!
# ler dado
def get_data(point):
    data = read_exported_series(point)

    lat = data['lat'][0]
    lon = data['lon'][0]

    data_low = filtro.filtra_dados(data['ssh'], data.index, 'low', modelo = False)
    # mudar aqui o valor inicial pra pegar diferentes horas
    data_low = data_low[9::24]
    data_high = filtro.filtra_dados(data_low, data.index[::24], 'high')

    data_filt = data_high

    return data, lat, lon, data_filt


def get_data_stats(series, place):
    gen_stats_point = stats.general_stats(series)
    gen_stat_df = pd.DataFrame(gen_stats_point, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)
    os.makedirs(f'{fig_folder}/stats/{place}/', exist_ok=True)
    gen_stat_df.to_csv(f'{fig_folder}/stats/{place}/gen_{place}.csv')
    

def get_correlation_matrix(lat, lon, data, t0, tf, json_path):
    if os.path.exists(json_path):
        return json_path
    json_dict = {}

    for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
        print(model)
        # setando caminho do servidor
        reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
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


def get_reanalisys(point, model):
    point_info = json.load(open(f'/home/bcabral/mestrado/{point[:-4]}.json'))

    reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
    reanalisys = set_reanalisys_dims(reanalisys, model)
    latlon = point_info[model]
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
    mod_ssh = model_series['ssh'].values
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
    for point in os.listdir(data_path):
        if point[-1] != 'v':
            continue
        data, lat, lon, data_filt = get_data(data_path + point)
        # ver o que fazer quando eu  tiver nan -> ex imbituba

        gplots.plot_time_series(data['ssh'], 'Série Temporal de ' + point[:-4], f'{fig_folder}/ponto_serie/', point[:-4])
        # nomear eixos (cm e data)
        gplots.plot_spectrum(data['ssh'], f'Espectro de {point[:-4]}', f'{fig_folder}/spectra/{point[:-4]}/', f'spec_{point[:-4]}')
        # tirar label
        gplots.plot_double_spectrum(data['ssh'], data_filt, f'Medido vs Filtrado ({point[:-4]})', f'{fig_folder}/spectra/{point[:-4]}/', f'comp_spec_{point[:-4]}')

        get_correlation_matrix(lat, lon, data_filt, data.index[0], data.index[-1], f'/home/bcabral/mestrado/{point[:-4]}.json')
        get_data_stats(data_filt, point)
        for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
            reanalisys, fil_reanalisys, filt_time = get_reanalisys(point, model)
            reanalisys = reanalisys * 100 # pasando pra cm
            fil_reanalisys = fil_reanalisys * 100 # passando pra cm

            gplots.plot_spectrum(reanalisys, f'Espectro de {point[:-4]} ({model})', f'{fig_folder}/spectra/{point[:-4]}/', f'spec_{model}', data = False)
            # ajustar a unidade acima
            gplots.plot_double_spectrum(reanalisys, fil_reanalisys, f'Original vs Filtrado ({point[:-4]} - {model})',
                                        f'{fig_folder}/spectra/{point[:-4]}/',f'comp_spec_{model}', mdata = False)
            
            # gplots.compare_spectra(data, reanalisys, f'Espectro {point[:-4]} vs {model}', f'{fig_folder}/spectra/{point[:-4]}',f'crosspec_{model}')
            gplots.compare_spectra(data_filt, reanalisys, f'Espectro Filtrado {point[:-4]} vs {model}', f'{fig_folder}/spectra/{point[:-4]}/',f'crosspec_filt_{model}')

            
            gplots.compare_time_series(data_filt, fil_reanalisys, f'{point[:-4]} vs {model}', f'{fig_folder}/series/{point[:-4]}/',f'{model}')
            # faltam labels e botar na mesma unidade o modelo e o ponto

            get_reanalisys_stats(data_filt, fil_reanalisys, point[:-4], model)
