import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from plota_dados_marinha import get_lat_lon, nome_estacao

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from sys import path


def find_point(name):
    pontos = {
        'cananeia' : [-25.02, -47.93],
        'fortaleza' : [-3.72, -38.47],
        'ilha_fiscal' : [-22.90, -43.17],
        'imbituba' : [-28.13, -48.40],
        'macae' : [-22.23, -41.47],
        'rio_grande' : [-32.13, -52.10],
        'salvador' : [-12.97, -38.52],
        'santana' : [-0.06, -51.17],
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
    
    for ponto in pontos:
        if ponto.lower() in name.lower():
            return pontos[ponto]


def read_gloss(path, name):
    ponto = find_point(name)
    serie = pd.read_csv(path)

    if len(serie.columns) == 5:
        serie.loc[len(serie)] = serie.columns.to_list()
        serie.columns=['ano', 'mes', 'dia', 'hora', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:00:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]
        
    elif len(serie.columns) == 6:
        serie.columns=['ano', 'mes', 'dia', 'hora', 'minuto', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            minu = serie ['minuto'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora', 'minuto'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]

    elif len(serie.columns) == 7:
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
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]


    serie['ssh'] = serie['ssh'].astype(float)/10
    serie = serie.mask(serie < -1000)
    return serie

def read_marinha(name):
    path = f'/Users/breno/Downloads/Dados/{name}'
    
    df = pd.read_csv(path, skiprows=11, sep=';',encoding='latin-1')
    df.loc[len(df)] = df.columns.to_list()
    df.columns = ['data', 'ssh', 'Unnamed: 2']
    df = df.drop(columns=['Unnamed: 2'])

    # Converter 'data' para formato datetime, se necessário
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M')
    df.set_index('data', inplace=True)

    df['ssh'] = df['ssh'].astype(int)

    df = df.sort_index()

    lat_lon = get_lat_lon(name)
    df['lat'] = lat_lon[0]
    df['lon'] = lat_lon[1]


    return df

def get_simcosta_row(n, name):
    df = pd.read_csv(f'/Users/breno/Documents/Mestrado/dados/simcosta/{name}', skiprows=n).dropna()
    return df

def read_simcosta(name):
    path = f'/Users/breno/Documents/Mestrado/dados/simcosta/{name}'
    n = 10
    while n < 35:
        try:
            df = get_simcosta_row(n, name)
            if df.columns[0].upper() == 'YEAR':
                break
            n+=1

        except Exception as e:
            n += 1
            
    return df

def plota_pontos(pts):
    coord = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    ax.set_extent([-55, -30, 5, -30], crs=ccrs.PlateCarree())

    # ax.set_extent([-40, 10, -50, -30], crs=coord)

    for point in pts:
        plt.plot(point[1], point[0],
                color='red', linewidth=2, marker='P',
                transform=ccrs.PlateCarree()
                )  
    lons = np.arange(-70, -20, 5)
    lats = np.arange(-25, 10, 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.tight_layout()
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/nne/mapa_estacoes.png')

files = {}

for i in os.listdir('/Volumes/BRENO_HD/GLOSS'):
    if i[-3:] != 'csv' or i == 'PedroPaulo_rocks_UHSLC.csv' or i[0] =='.':
        continue
    files[i+'g1'] = read_gloss('/Users/breno/Documents/Mestrado/dados/gloos/'+ i, i)

for j in os.listdir('/Users/breno/Documents/Mestrado/dados/gloos/goos'):
    files[j+'g2'] = read_gloss('/Users/breno/Documents/Mestrado/dados/gloos/goos/'+ j, j)

# resolucao diaria
# for k in os.listdir('/Users/breno/Documents/Mestrado/dados/gloos/havai'):
#     files[k+'g3'] = read_gloss('/Users/breno/Documents/Mestrado/dados/gloos/havai/'+ k, k)

for l in os.listdir('/Users/breno/Downloads/Dados'):
    files[nome_estacao(l)[:-1]+'m'] = read_marinha(l)

for m in os.listdir('/Users/breno/Documents/Mestrado/dados/simcosta'):
    if m[-3:] != 'csv':
        continue
        # imbituba
        # guaratuba
    files[m+'s'] = read_simcosta(m)

