# feito pro servidor do LOF

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib.ticker as mticker
import os
import datetime
# from le_reanalise import get_lat_lon, set_reanalisys_dims
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# acho que vou pegar a resolucao de cada modelo e pegar uma malha de 9 pontos pra cada.

home_path =  '/home/bcabral/'
# botar pra rodar toda essa brincadeira aqui pra todos os modelos!

def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude', 'surf_el':'ssh'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude', 'Time':'time', 'eta_t':'ssh'})
    else:
    	for i in list(reanalisys.variables):
            default = ['latitude', 'longitude', 'time']
            if i not in default:
                reanalisys = reanalisys.rename({i:'ssh'})

    return reanalisys


def get_lat_lon(reanalisys):
    return (reanalisys.latitude.values, reanalisys.longitude.values)


def plot_grid_point(lat, lon, nome_modelo, nome_ponto, figs_folder, lons, lats, lat_mod, lon_mod):
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lon_max, lon_min, lat_max, lat_min = lon + 1 , lon - 1, lat + 1, lat - 1
    ax.set_extent([lon_max, lon_min, lat_max, lat_min], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='0.3')
    ax.add_feature(cfeature.LAKES, alpha=0.9)  
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',  name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', zorder=10) 


    plt.plot(lon, lat,
            color='red', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            )  
    plt.text(lon - 0.05, lat - 0.05, nome_ponto,
          horizontalalignment='right', color = 'red', weight = 'bold',
          transform=ccrs.PlateCarree())  

    
    plt.plot(lon_mod, lat_mod,
            color='green', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            )    

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                      color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    ax.set_title(f'Ponto {nome_ponto}')

    plt.tight_layout()
    plt.savefig(figs_folder + f'{nome_modelo}_{nome_ponto}_dominio.png')


pontos_dado = {
    'cananeia' : [-25.02, -47.93],
    'fortaleza' : [-3.72, -38.47],
    'ilha_fiscal' : [-22.90, -43.17],
    'imbituba' : [-28.13, -48.40],
    'macae' : [-22.23, -41.47],
    'rio_grande' : [-32.13, -52.10],
    'salvador' : [-12.97, -38.52],
    # 'santana' : [-0.06, -51.17],
    #'São Pedro e São Paulo' : [3.83, -32.40],
    'ubatuba' : [-23.50, -45.12],
    'rio_grande2' : [-32.17, -52.09], #RS
    'tramandai' : [-30.00, -50.13], #RS
    'paranagua' : [-25.50, -48.53], #PR
    'pontal_sul' : [-25.55, -48.37], #PR
    'ilhabela' : [-23.77, -45.35], #SP
    'dhn' : [-22.88, -43.13],#RJ
    'ribamar': [-2.56, -44.05]#MA
    }

model_path = '/data3/MOVAR/modelos/REANALISES'
modelos = ['BRAN', 'CGLO', 'ECCO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  ,'SODA']
results = {}
for model in modelos:
    i = datetime.datetime.now()
    results[model] = {}
    path_model = f'/data3/MOVAR/modelos/REANALISES/{model}/SSH/2014/'
    reanalisys = xr.open_mfdataset(path_model + '*.nc')
    reanalisys = set_reanalisys_dims(reanalisys, model)
    lat_resolution = abs(float(reanalisys.latitude[0] - reanalisys.latitude[1]))
    lon_resolution = abs(float(reanalisys.longitude[0] - reanalisys.longitude[1]))
    # PEGANDO A GRADE
    for ponto in pontos_dado:
     # FYI esse bloco leva 31 segundos por loop (por ponto). como sao 15 pts, leva quase 8 		#   min pra terminar de rodar tudo (por modelo)
        print('comecando ' + ponto)
        results[model][ponto] = []
        i = datetime.datetime.now()
     # botar de um jeito que serie com NaN seja ignorada!
        series ={
    	'A': reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'B' : reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'C' :  reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time'),
    	'D' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'E' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'F' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time'),
    	'G' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'H' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'I' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time')
    	}
    	
    	# tem que importar a itertools
        combinacoes = list(itertools.combinations(series.keys(), 2))
        for serie1, serie2 in combinacoes:
            try:
                ks_statistic, p_value = ks_2samp(series[serie1]['ssh'], series[serie2]['ssh'])
                print(f"KS Statistic between {serie1} and {serie2}: {ks_statistic}, p-value: {p_value}")
                results[model][ponto].append((serie1, serie2, ks_statistic, p_value))
            except Exception:
                print("Serie com Nan " + serie1 +  serie2)
	    	
    	# TODO: analise dos pontos
        print(f'Tempo de execução {(datetime.datetime.now() - i).seconds} segundos')
	## Ver como vou fazer pra fazer essas diferencas
        
    print('____OK____' + model)
    print(f'Tempo de execução {(datetime.datetime.now() - i).seconds} segundos')


## SOLUCAO CHATGPT PRA PEGAR N PONTOS AO REDOR -> A CONFERIR E ADAPTAR
def extrair_pontos_ao_redor(ponto_central, lat_resolution, lon_resolution):
    # Extrair coordenadas do ponto central
    lat_central, lon_central = ponto_central
    
    # Calcular coordenadas dos pontos ao redor
    pontos_ao_redor = []
    for lat_diff in [-lat_resolution, 0, lat_resolution]:
        for lon_diff in [-lon_resolution, 0, lon_resolution]:
            novo_ponto = (lat_central + lat_diff, lon_central + lon_diff)
            pontos_ao_redor.append(novo_ponto)
    
    return pontos_ao_redor

# Exemplo de uso
ponto_central = (40.0, -74.0)  # Exemplo de ponto central
lat_resolution = 0.1  # Resolução de latitude
lon_resolution = 0.1  # Resolução de longitude

pontos_ao_redor = extrair_pontos_ao_redor(ponto_central, lat_resolution, lon_resolution)
print("Pontos ao redor do ponto central:", pontos_ao_redor)
##### OUTRA ALTERNATIVA DO CHATGPT

series = {}
# Let's assume 'reanalisys' is your dataset and 'pontos_dado' is a dictionary containing points data

# Lista de labels para os pontos
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']

for label in labels:
    lat, lon = pontos_dado[ponto][0], pontos_dado[ponto][1]  # Coordenadas do ponto central
    if label == 'A':
        lat += lat_resolution
        lon -= lon_resolution
    elif label == 'B':
        lat += lat_resolution
    elif label == 'C':
        lat += lat_resolution
        lon += lon_resolution
    elif label == 'D':
        lon -= lon_resolution
    elif label == 'E':
        # Ponto central, já adicionado anteriormente
        pass
    elif label == 'F':
        lon += lon_resolution
    elif label == 'G':
        lat -= lat_resolution
        lon -= lon_resolution
    elif label == 'H':
        lat -= lat_resolution
    elif label == 'I':
        lat -= lat_resolution
        lon += lon_resolution
    
    # Seleção dos dados com a vizinhança e remoção dos valores NaN
    selected_data = reanalisys.sel(latitude=lat, longitude=lon, method='nearest').dropna('time')
    
    # Adiciona ao dicionário apenas se houver dados após a remoção dos NaN
    if not selected_data.empty:
        series[label] = selected_data


### End ###
