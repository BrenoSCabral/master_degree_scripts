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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

import sys

sys.path.append(
    '../'
)
import model_filt
import specs_amp
from read_data import read_exported_series, treat_exported_series, all_series, sep_series
from read_reanalisys import set_reanalisys_dims
from model_data import get_model_region, get_correlation, mapa_corr
import filtro
import get_skill





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
    os.makedirs(f'/home/bcabral/mestrado/fig/comp_series3/{place}/stats/', exist_ok=True)
    
    gen_stat_df.to_csv(f'/home/bcabral/mestrado/fig/comp_series3/{place}/stats/gen_{place}.csv')


def get_reanalisys_stats(data, reanalisys, place, model, folder):
    stat_met = stats(reanalisys, data)
    dep_stat_met = dependent_stats(reanalisys, data)
    gen_stats = general_stats(reanalisys)

    # gen_stats['Dado'] = general_stats(data)

    stat_df = pd.DataFrame(stat_met, index=['CORR', 'NBIAS', 'NRMSE', 	'SI']).round(2)
    dep_stat_df = pd.DataFrame(dep_stat_met, index=['BIAS', 	'RMSE','SCRMSE']).round(2)
    gen_stat_df = pd.DataFrame(gen_stats, index=['SOMA', 'MÉDIA', 	'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)

    # TODO: acho que seria mais legal juntar todas essas metricas em um so arquivo. Mais facil mexer depois
    os.makedirs(f'{folder}/{place}/stats/', exist_ok=True)
    stat_df.to_csv(f'{folder}/{place}/stats/comp_{model}.csv')
    dep_stat_df.to_csv(f'{folder}/{place}/stats/dep_{model}.csv')
    gen_stat_df.to_csv(f'{folder}/{place}/stats/gen_{model}.csv')



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


def check_reanal(reanal, point_info, data, model):
    reanalisys = xr.concat(list(reanal.values()), dim="time")
    latlon = point_info[model]
    model_series = reanalisys.sel(latitude=latlon[0], longitude=latlon[1], method='nearest')
    model_series = model_series.sel(time=slice(t0, tf + pd.Timedelta(days=1)))

    full_time_index = pd.date_range(start=t0, end=tf, freq='D')  # Diário, 00h
    filtered_reanal = model_filt.filtra_reanalise(model_series)


    # Normaliza o índice de tempo dos dados para 00h
    normalized_time = filtered_reanal['time'].to_index().normalize()
    filtered_reanal_normalized = filtered_reanal.assign_coords(time=filtered_reanal['time'].to_index().normalize())

    # Encontra os dias faltantes
    missing_days = full_time_index.difference(normalized_time)
    # Cria uma lista para armazenar os dados substitutos
    substitute_data = []

    # feito especialmente pros dias do hycom
    for missing_day in missing_days:
        # Define o dia anterior às 21h
        for h in range(1,24):
            if model == 'BRAN' and h == 12:
                continue
            previous_day = missing_day - pd.Timedelta(days=1) + pd.Timedelta(hours=h)
            # Verifica se há dados no dia anterior às 21h
            if previous_day in filtered_reanal['time'].to_index():
                print(f"Dado encontrado para o dia {missing_day} no horário {previous_day}")
                # Extrai os dados do dia anterior às 21h
                substitute_data.append(filtered_reanal.sel(time=previous_day).assign_coords(time=missing_day))
                break
            
            if h ==23:
                print(f"Nenhum dado encontrado para o dia {missing_day} ou no horário {previous_day}")

    complete_data = filtered_reanal_normalized.sel(time=~normalized_time.isin(missing_days))

    # Adiciona os dados substitutos (dias anteriores)
    if substitute_data:
        complete_data = xr.concat([complete_data] + substitute_data, dim='time')

    # Ordena por tempo (caso necessário)
    complete_data = complete_data.sortby('time')



    # dropar o modelo nos momentos em que eu nao tiver dado
    ind_reanal = complete_data['time'].to_index()
    ind_data = data.index

    common_dates = ind_data.intersection(ind_reanal)
    if complete_data['time'].to_index().duplicated().sum() >0:
        complete_data = complete_data.sel(time=~complete_data['time'].to_index().duplicated())

    filtered_reanal_common = complete_data.sel(time=common_dates)

    # cortando caso um seja maior que o outro
    if len(filtered_reanal_common.values) > len(data):
        print(f'{model} maior do que os DADOS')
        tf2 = tf - datetime.timedelta(days=len(filtered_reanal_common.values) - len(data))
        filtered_reanal_common = filtered_reanal_common.sel(time=slice(t0, tf2))



    return filtered_reanal_common


def plot_mapa(latlon_point, latlon_model, name_point, name_model):
    """
    Plota o ponto de um dado observado e o ponto de um modelo em um mapa com zoom de 5 graus.

    Parâmetros:
    -----------
    latlon_point : tuple
        Tupla com as coordenadas (latitude, longitude) do ponto observado.
    latlon_model : tuple
        Tupla com as coordenadas (latitude, longitude) do ponto do modelo.
    name_point : str
        Nome do ponto observado.
    name_model : str
        Nome do ponto do modelo.
    """
    # Extrai as coordenadas
    lat_point, lon_point = latlon_point
    lat_model, lon_model = latlon_model

    # Cria a figura e o eixo com projeção de mapa
    fig, ax = plt.subplots(figsize=(10, 6), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

    # Define o zoom de 5 graus em torno do ponto de análise
    zoom = 1  # Graus de zoom
    ax.set_extent([lon_point - zoom, lon_point + zoom, 
                   lat_point - zoom, lat_point + zoom], crs=ccrs.PlateCarree())

    # Adiciona características do mapa
    ax.add_feature(cfeature.LAND, edgecolor='black')  # Adiciona continentes
    ax.add_feature(cfeature.OCEAN)                   # Adiciona oceanos
    ax.add_feature(cfeature.COASTLINE)               # Adiciona linha costeira
    ax.add_feature(cfeature.BORDERS, linestyle=':')  # Adiciona fronteiras
    ax.add_feature(cfeature.LAKES, alpha=0.5)        # Adiciona lagos
    ax.add_feature(cfeature.RIVERS)                  # Adiciona rios

    # Adiciona gridlines
    ax.gridlines(draw_labels=True, linestyle='--', color='gray')

    # Plota o ponto observado
    ax.scatter(lon_point, lat_point, color='red', label=name_point, s=100, edgecolor='black', transform=ccrs.PlateCarree())

    # Plota o ponto do modelo
    ax.scatter(lon_model, lat_model, color='blue', label=name_model, s=100, edgecolor='black', transform=ccrs.PlateCarree())

    # Adiciona título e legenda
    ax.set_title(f'Localização de {name_point} (Observado) e {name_model} (Modelo)')
    ax.legend(loc='upper right')

    os.makedirs(f'/home/bcabral/mestrado/fig/comp_series3/{name_point}/', exist_ok=True)
    plt.savefig(f'/home/bcabral/mestrado/fig/comp_series3/{name_point}/map_{name_model}.png', dpi=200)


# def plot_series(spaos, data, data_name):
#     for year in data.index.year.unique():
#         fig = plt.figure(figsize=(14,8))
#         ax = fig.add_subplot(111)
#         ax.spines['top'].set_color('none')
#         ax.spines['bottom'].set_color('none')
#         ax.spines['left'].set_color('none')
#         ax.spines['right'].set_color('none')
#         ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

#         ax1= fig.add_subplot(711)
#         ax2= fig.add_subplot(712)
#         ax3= fig.add_subplot(713)
#         ax4= fig.add_subplot(714)
#         ax5= fig.add_subplot(715)
#         ax6= fig.add_subplot(716)
#         ax7= fig.add_subplot(717)

#         axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

#         time = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')

#         for spao, axe in zip(spaos, axes):
#             axe.plot(time, spaos[spao]*100, label=spao)
#             axe.plot(time, data, label= data_name + ' Diário')
#             axe.grid()
#             axe.legend(fontsize='x-small', loc='lower left')
#             axe.set_ylabel(spao)

#         ax.set_ylabel('(cm)')
#         # plt.show()
#         plt.tight_layout()

#         os.makedirs(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/', exist_ok=True)
#         plt.savefig(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/{year}.png', dpi=200)


# import os
# import pandas as pd
# import matplotlib.pyplot as plt

def plot_series(spaos, data, data_name):
    # Itera sobre cada ano único presente na coordenada de tempo de `data`
    for year in data.index.year.unique():
        # Cria o range de tempo para o ano atual (01/01 a 31/12)
        time = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')

        # Filtra os dados para o ano atual
        data_year = data[data.index.year == year]

        # Filtra os dados de `spaos` para o ano atual
        spaos_year = {spao: spaos[spao].sel(time=spaos[spao]['time'].dt.year == year) for spao in spaos}

        # Cria a figura e o eixo principal
        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        # Cria subplots para cada SPOA
        ax1 = fig.add_subplot(711)
        ax2 = fig.add_subplot(712)
        ax3 = fig.add_subplot(713)
        ax4 = fig.add_subplot(714)
        ax5 = fig.add_subplot(715)
        ax6 = fig.add_subplot(716)
        ax7 = fig.add_subplot(717)

        axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]

        # Plota os dados para cada SPOA
        for spao, axe in zip(spaos_year, axes):
            # Verifica se há dados para o SPOA no ano atual
            if not spaos_year[spao].isnull().all():
                # Reindexa os dados para o ano completo
                spaos_reindexed = spaos_year[spao].reindex(time=time, fill_value=np.nan)

                # Plota os dados
                axe.plot(time, spaos_reindexed * 100, label='SPAOs')
                axe.plot(time, data_year.reindex(time), label=f'{data_name.capitalize().replace("_", " ")}')
                axe.grid()
                axe.yaxis.set_label_position("right")
                axe.set_ylabel(spao)

            else:
                print(f"Nenhum dado disponível para {spao} no ano {year}.")


        # Configurações finais
        ax.set_ylabel('(cm)')
            # Adiciona uma legenda única fora dos eixos
        handles, labels = [], []
        for axe in axes:
            h, l = axe.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        # Remove duplicatas
        unique_labels, unique_handles = [], []
        for label, handle in zip(labels, handles):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)
        # Adiciona a legenda fora dos eixos
        fig.legend(unique_handles, unique_labels, loc='lower center', frameon=False, ncol=2)
        plt.tight_layout()

        # Salva a figura
        os.makedirs(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/', exist_ok=True)
        plt.savefig(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/{year}.png', dpi=200)
        plt.close()  # Fecha a figura para liberar memória



def plot_heat(data, spaos, data_name):

    plt.figure(figsize=(14, 8))

    # A. Gráfico de Dispersão com Amostragem
    observed = np.asarray(data['ssh'])
    for model in spaos:
        plt.scatter(
            np.asarray(data['ssh']),
            np.asarray(spaos[model]) * 100,
            alpha=0.4,
            s=20,
            label=f"{model} (Skill={get_skill.ss2(np.asarray(data['ssh']),np.asarray(spaos[model]) *100):.2f})"
        )

    # B. Linha de Referência 1:1
    lims = [np.min(observed), np.max(observed)]
    plt.plot(lims, lims, 'k--', alpha=0.5)

    # C. Customização
    plt.xlabel('Observation Values', fontsize=12)
    plt.ylabel('Model Values', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.2)


    plt.tight_layout()
    os.makedirs(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/', exist_ok=True)
    plt.savefig(f'/home/bcabral/mestrado/fig/comp_series3/{data_name}/heat_map.png', dpi=200)
    plt.close()  # Fecha a figura para liberar memória


all_data = get_all_available_data() # TODO: Precisa mudar aqui de H pra h (deprecated)

series = join_series(all_data)
series = {clean_key(key): value for key, value in series.items()}

# pegar as series
# filtrar as serie
# compor o merged
fseries = join_filt_series(all_data)
fseries = {clean_key(key): value for key, value in fseries.items()}

# pts = []
# for point in list(series.keys()):
#     if point == 'CANIVETE' or point == 'CAPITANIA DE SALVADOR' or point == 'CIA DOCAS DO PORTO DE SANTANA' \
#         or point == 'IGARAPÉ GRD DO CURUÁ' or point == 'PAGA DÍVIDAS I' or point == 'PORTO DE VILA DO CONDE' \
#         or point == 'Salvador_glossbrasil' or point == 'SERRARIA RIO MATAPI':
#         continue
#     pts.append(point)


for point in fseries:
    if point == 'CANIVETE' or point == 'CIA DOCAS DO PORTO DE SANTANA' \
        or point == 'IGARAPÉ GRD DO CURUÁ' or point == 'PAGA DÍVIDAS I' or point == 'PORTO DE VILA DO CONDE' \
        or point == 'Salvador_glossbrasil' or point == 'SERRARIA RIO MATAPI' or point=='salvador':
        continue

    print(point)

    os.makedirs(f'/home/bcabral/mestrado/fig/spec_04', exist_ok=True)

    specs_amp.spec_anal(np.asarray(fseries[point]['ssh']),
              f'/home/bcabral/mestrado/fig/spec_04/{point}_filt.png', False)
    specs_amp.spec_anal(np.asarray(series[point]['ssh']),
              f'/home/bcabral/mestrado/fig/spec_04/{point}_brute.png', True)
    spaos = {}
    # tirando pontos dentro de estuario ou com series esquisitas

    data = fseries[point]
    years = range(data.index[0].year, data.index[-1].year +1)
    t0, tf = data.index[0], data.index[-1]
    lat = series[point]['lat'][0]
    lon = series[point]['lon'][0]

    # esse loop abaixo o ponto indicado pelo json como maior correlacao para a reanalise
    get_data_stats(data['ssh'], point)

    try:
        for model in ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']:
            print(f'{model}')
            # plotar a serie temporal
            # plotar o mapa do ponto do modelo vs o ponto do dado
            point_info = json.load(open(f'/home/bcabral/mestrado/{point}_25pts.json'))
            reanal = {}
            for year in years:
                reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                                    , model)

            filtered_reanal_common = check_reanal(reanal, point_info, data, model)
            spaos[model] = filtered_reanal_common


            # agora basta fazer os plots
            print('pegnado stats')
            get_reanalisys_stats(data['ssh'], filtered_reanal_common.values * 100, point, model, '/home/bcabral/mestrado/fig/comp_series4/')
            get_skill.run_all(data=data['ssh'], model=filtered_reanal_common.values*100,
                              path=f'/home/bcabral/mestrado/fig/comp_series4/{point}/stats/{model}')



            print('plotando mapa')
            plot_mapa((series[point]['lat'].iloc[0], series[point]['lon'].iloc[0]),
                        (point_info[model][0], point_info[model][1]), point, model)
            
            specs_amp.double_spec_anal(np.asarray(series[point]['ssh']), np.asarray(filtered_reanal_common)*100,
                                       f'/home/bcabral/mestrado/fig/comp_series4/{point}/spec_{model}',
                                       point, model)



        spaos['GOFS'] = spaos.pop('HYCOM')
        plot_series(spaos, data, point)
        plot_heat(data, spaos, point)
    except:
        print(f'Não conseguiu fazer - {point}')

print('\n\nTERMINOUU')
