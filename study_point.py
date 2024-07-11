from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import numpy as np
import pandas as pd
import os
from read_reanalisys import set_reanalisys_dims
from read_data import read_exported_series, treat_exported_series
import filtro
import json
from sys import path
path.append(
    'old'
)
import stats
from scipy.fft import fft, fftfreq


def study_points():
    model_path = '/data3/MOVAR/modelos/REANALISES/'
    data_path = '/home/bcabral/mestrado/data/'
    fig_folder = '/home/bcabral/mestrado/fig/'
    year = 2012

    places = ['CAPITANIA DE SALVADORm_2012.csv', 'Fortaleza_2012.csv', 'ilha_fiscal_2012.csv', 'TEPORTIm_2012.csv']
    models = ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']

    for place in places:
        print(f'rodando para {place}')
        data = read_exported_series(data_path + str(year) + '/' + place)
        data_low = filtro.filtra_dados(data['ssh'], data.index, 'low', modelo = False)
        data_low = data_low[::24]
        data_high = filtro.filtra_dados(data_low, data.index[::24], 'high')
        data_filt = data_high

        point_info = json.load(open(f'/home/bcabral/mestrado/{place[:-4]}.json'))
        gen_stats_point = stats.general_stats(data_filt)
        gen_stat_df = pd.DataFrame(gen_stats_point, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)
        os.makedirs(f'{fig_folder}/stats/{place}/cut/', exist_ok=True)
        gen_stat_df.to_csv(f'{fig_folder}/stats/{place}/gen_{place}.csv')

        make_point_spectrum(data['ssh'], data_filt, place[:-4])


        for model in models:
            reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
            reanalisys = set_reanalisys_dims(reanalisys, model)
            latlon = point_info[model]
            model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
            mod_ssh = model_series['ssh'].values
            mod_time = model_series['time'].values
            mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)
            # mudancas abaixo pro ponto que comeca em junho
            plot_series(data_filt, mod_band[152:], mod_time[152:], place, model, fig_folder)
            do_stats(data_filt, mod_band[152:], place, model, fig_folder)
            make_model_spectrum(mod_ssh, mod_band, place[:-4], model)


def plot_series(data, model, time, name_data, name_model, fig_path):
    fig = plt.figure(figsize=(14,8))
    data = data[50:]
    model = model[50:]
    time = time[50:]
    if name_data == 'Fortaleza_2012.csv':
        data = data*100
    plt.plot(time, model*100, label = name_model)
    plt.plot(time, data, label = name_data)
    import datetime
    plt.xlim([datetime.date(2012, 1, 1), datetime.date(2013, 1, 1)])
    plt.grid()
    plt.legend(fontsize='x-small', loc='lower left')
    os.makedirs(f'{fig_path}/serie/{name_data}/cut/', exist_ok=True)
    plt.savefig(f'{fig_path}/serie/{name_data}/cut/{name_model}.png')


def do_stats(data, model, data_name, model_name, out_path):
    data = data[50:]
    model = model[50:]
    stat_met = stats.stats(model * 100, data)
    dep_stat_met = stats.dependent_stats(model * 100, 		data)
    gen_stats = stats.general_stats(model * 100)

    # gen_stats['Dado'] = stats.general_stats(data)

    stat_df = pd.DataFrame(stat_met, index=['CORR', 'NBIAS', 'NRMSE', 	'SI']).round(2)
    dep_stat_df = pd.DataFrame(dep_stat_met, index=['BIAS', 	'RMSE','SCRMSE']).round(2)
    gen_stat_df = pd.DataFrame(gen_stats, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)

    stat_df.to_csv(f'{out_path}/stats/{data_name}/cut/comp_{model_name}.csv')
    dep_stat_df.to_csv(f'{out_path}/stats/{data_name}/cut/dep_{model_name}.csv')
    gen_stat_df.to_csv(f'{out_path}/stats/{data_name}/cut/gen_{model_name}.csv')


def make_point_spectrum(data, filtered, name):
    data = np.asarray(data)


    fig = plt.figure(figsize=(14,8))
    fig.suptitle(f'Serie medida vs filtrada ({name})')
    ax = fig.add_subplot(111)

    N1 = len(data) # numero de medicoes
    T1 = 1.0/24 # frequencia das medicoes (nesse caso = 1 medicao por h)
    yif1 = fft(data)
    xif1 = fftfreq(N1,T1)
    tif1 = 1/xif1


    N = len(filtered) # numero de medicoes
    T = 1.0 # frequencia das medicoes (nesse caso = 1 medicao a cada 24h)
    yif = fft(filtered)
    xif = fftfreq(N,T)
    tif = 1/xif

    ax.semilogx(tif1, 2.0/N1 * np.abs(yif1), label = 'Medida')
    ax.semilogx(tif, 2.0/N * np.abs(yif), label = 'Filtrado')

    # ax.set_xlim(10,20)
    # ax.set_xticks([3, 5, 10, 20, 30, 40]) 

    # ax.set_xticklabels([])
    ax.legend()
    # ax.set_yticks([])

    # ax.ylim([-0.1, 40])
    ax.grid()

    # ax.set_xticklabels([3, 5, 10, 20, 30, 40])

    ax.set_ylabel('Densidade Espectral [m²/dia]')
    ax.set_xlabel('Dias')
    plt.savefig('/Users/breno/Documents/Mestrado/resultados/2012/figs/compara_espectro_{name}')


def make_model_spectrum(model, filtered, point, model_name):
    model = np.asarray(model)


    fig = plt.figure(figsize=(14,8))
    fig.suptitle(f'Espectro {point} - {model_name}')
    ax = fig.add_subplot(111)

    N1 = len(model) # numero de medicoes
    T1 = 1.0 # frequencia das medicoes (nesse caso = 1 medicao por h)
    yif1 = fft(model)
    xif1 = fftfreq(N1,T1)
    tif1 = 1/xif1

    yif = fft(filtered)
    xif = fftfreq(N1,T1)
    tif = 1/xif

    ax.semilogx(tif1, 2.0/N1 * np.abs(yif1), label = 'Medida')
    ax.semilogx(tif, 2.0/N1 * np.abs(yif), label = 'Filtrado')

    # ax.set_xlim(10,20)
    # ax.set_xticks([3, 5, 10, 20, 30, 40]) 

    # ax.set_xticklabels([])
    ax.legend()
    # ax.set_yticks([])

    # ax.ylim([-0.1, 40])
    ax.grid()

    # ax.set_xticklabels([3, 5, 10, 20, 30, 40])

    ax.set_ylabel('Densidade Espectral [m²/dia]')
    ax.set_xlabel('Dias')
    plt.savefig('/Users/breno/Documents/Mestrado/resultados/2012/figs/espectro/{point}/espectro_{model_name}')


'''
TODO: 
- Plotar todas as series temporais (pode ser brutas mesmo) pra ver em que unidade elas estao (comparar com ilha fiscal que eu sei qual unidade ta)
- Rodar o codigo pra fazer as estatisticas e os plots
- Montar um diagramazinho pra dar uma olhada na propagação da onda
- seria interessante tentar ver a coerencia do sinal (apesar de eu achar isso improvavel de se fazer de hj pra amanha)
'''



def get_model_region(lat, lon, model_name, reanalisys):
    # reanalisys = xr.open_mfdataset(model_path + '/*SSH*.nc')
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


# salva figura
    # os.makedirs(f'{fig_folder}/correlacao/{nome_ponto}/', exist_ok=True)
    # plt.savefig(f'{fig_folder}/correlacao/{nome_ponto}/{nome_modelo}.png')


def get_corr(data_name, server, year):


    data_low = filtro.filtra_dados(data['ssh'], data.index, 'low', modelo = False)
    data_low = data_low[::24]
    data_high = filtro.filtra_dados(data_low, data.index[::24], 'high')

    data_filt = data_high

    # ECCO atualiza de 3 em 3 dias
    models = ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  'ECCO' ,'SODA']
    json_dict = {}
    for model in models:
        print(model)
        # setando caminho do servidor
        reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
    #model = 'BRAN'
        reanal_subset = get_model_region(lat, lon, model, reanalisys)
        reanal_subset['ssh'].load()
        reanal_subset = reanal_subset.sel(time=slice(data.index[0], data.index[-1]))


        correlation, latlons= get_correlation(data_filt, reanal_subset, fig_folder)

        correlation[np.isnan(correlation)] = -np.inf
        max_index = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        json_dict[model] = latlons[max_index]
        # mapa_corr(lon, lat, reanal_subset, correlation, model, '_'.join(str.split(data_name, '_')[:-1]), fig_folder)
    
    with open(f'/home/bcabral/mestrado/{data_name[:-4]}.json', 'w') as f:
        json.dump(json_dict, f)


def main():
    print('rodando main')

    # places = ['Santana_2014.csv', 'Fortaleza_2014.csv', 'salvador2_2014.csv',
    #         'macae_2014.csv', 'ilha_fiscal_2014.csv', 'ubatuba_2014.csv', 'cananeia_2014.csv', 'imbituba_2014.csv']

    places = ['CAPITANIA DE SALVADORm_2012.csv', 'Fortaleza_2012.csv', 'ilha_fiscal_2012.csv', 'TEPORTIm_2012.csv']# , 'macae_2012.csv']

    for place in places:

        print(f'rodando para {place}')
        get_corr(place, True, 2012)

        # data_raw = read_exported_series(f'/Users/breno/Documents/Mestrado/resultados/2014/data/{place}')
        # data = treat_exported_series(data_raw)

        # plt.figure(figsize=(15,10))
        # plt.plot(data['ssh'])
        # plt.grid()
        # plt.savefig('/Users/breno/Documents/Mestrado/resultados/2014/figs/series/' + str.split(place, '.')[0] + '.png')

if __name__ == "__main__":
    main()

