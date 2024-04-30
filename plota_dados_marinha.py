import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


# Lendo os dados
path = '/Users/breno/Downloads/Dados'

def plota_pontos(pts):
    coord = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    # ax.set_extent([-40, 10, -50, -30], crs=coord)

    for point in pts:
        plt.plot(point[1], point[0],
                color='red', linewidth=2, marker='P',
                transform=ccrs.PlateCarree()
                )  
    lons = np.arange(-70, -20, 5)
    lats = np.arange(-50, 10, 5)
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
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/marinha/mapa_estacoes.png')

def convert_coord(coordinate):
    graus, minutos, segundos, direcao = coordinate
    decimal = graus + (minutos / 60) + (segundos / 3600)
    if direcao.upper() == 'W' or direcao.upper() == 'S':
        decimal *= -1  
    return decimal

def get_lat_lon(i):
    with open(path + '/' + i, 'r', encoding='latin-1') as arquivo:
        lat_brute = arquivo.readlines()[3]
        lat = (int(lat_brute[9:11]), int(lat_brute[13:15]), int(lat_brute[17:18]), lat_brute[19:20])

    with open(path + '/' + i, 'r', encoding='latin-1') as arquivo:
        lon_brute = arquivo.readlines()[4]
        lon = (int(lon_brute[10:13]), int(lon_brute[15:17]), int(lon_brute[19:20]), lon_brute[21:22])

    return(convert_coord(lat), convert_coord(lon))

def nome_estacao(i):
    with open(path + '/' + i, 'r', encoding='latin-1') as arquivo:
        return(arquivo.readlines()[2][13:])

def do_everything():
    pos = []
    norte = []
    for i in os.listdir(path):
        df = pd.read_csv(path + '/' + i, skiprows=11, sep=';',encoding='latin-1')
        df.loc[len(df)] = df.columns.to_list()
        df.columns = ['data', 'ssh', 'Unnamed: 2']
        df = df.drop(columns=['Unnamed: 2'])

        # Converter 'data' para formato datetime, se necessário
        df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M')
        df.set_index('data', inplace=True)

        df['ssh'] = df['ssh'].astype(int)

        df = df.sort_index()

        lat_lon = get_lat_lon(i)
        pos.append(lat_lon)
        if lat_lon[0] > -20:
            norte.append((df.index[0], df.index[-1], (df.index[-1] - df.index[0]).days, lat_lon[0])) 

        fig = plt.figure(figsize=(10, 5))
        plt.plot(df['ssh'])
        plt.title(nome_estacao(i))
        plt.grid()
        nans = df['ssh'].isna().sum()
        plt.annotate(f'No NaNs: {nans}', xy=(1, 1.03), xycoords='axes fraction', ha='right', va='top', fontsize=12, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
        plt.annotate(f'di: {df.index[0]}', xy=(0.2, 0.02), xycoords='axes fraction', ha='right', va='top', fontsize=6, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
        plt.annotate(f'df: {df.index[-1]}', xy=(1, 0.02), xycoords='axes fraction', ha='right', va='top', fontsize=6, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5'))
        plt.ylabel('SSH (cm)')

        plt.savefig(f'/Users/breno/Documents/Mestrado/dados/marinha/{nome_estacao(i)}.png')
        plt.close()

    plota_pontos(pos)





    fig, ax = plt.subplots()

    # Plotar as linhas horizontais para cada item da lista
    for i in norte:
        ax.hlines(y=i[-1], xmin=i[0], xmax=i[1], color='blue')
        # ax.text(data_inicio, i, nome, ha='right', va='center')  # Adiciona o nome do item próximo ao início da linha

    # Configurar o eixo y
    # ax.set_yticks(range(len(series_longas)))
    # ax.set_yticklabels([nome for nome, _, _ in series_longas])

    # Definir os rótulos dos eixos
    ax.set_xlabel('Data')

    # Definir o título do gráfico
    ax.set_title('Período de Tempo Séries Médias a Norte de 20o')

    # Rotacionar os rótulos do eixo x para melhorar a legibilidade
    plt.xticks(rotation=45)

    # Exibir o gráfico
    plt.tight_layout()
    plt.show()
