import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import itertools
import cartopy
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
import time
from scipy.fft import fft, fftfreq
path.append(
    '/Users/breno/Documents/Mestrado/tese/scripts'
)
import filtro
import le_dado
import le_reanalise

# faz a analise pro dado:

# fig_folder = '/Users/breno/Documents/Mestrado/tese/figs/rep3/'
# path_dado = '/Users/breno/dados_mestrado/dados/'
# path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR4/'

# # filename = 'Ilha Fiscal 2014.txt'
# # formato = 'csv'
# nome = 'Ilha Fiscal'

# data_lat = -22.90
# data_lon = -43.17

# data_lat = -2.56
# data_lon = -44.05

# pegando pontos pra fazer a perpendicular:

############ \\\\\\\\
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature



plt.rcParams.update({"font.size": 20})
SMALL_SIZE = 12
MEDIUM_SIZE = 22
LARGE_SIZE = 26
plt.rc("font", size=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)


def le_batimetria():
    bathy_file_path = Path('/Volumes/BRENO_HD/dados_mestrado/batimetria/gebco_bat.nc')

    bathy = xr.open_dataset(bathy_file_path)

    return bathy

def cut_bathy(bathy, data_lat, data_lon):
    lat_max = data_lat + 5
    lat_min = data_lat - 5
    lon_max =  data_lon + 5
    lon_min = data_lon -5
    bathy_cut = bathy.where((bathy.lat < lat_max) & 
                              (bathy.lon > lon_min) &
                              (bathy.lat > lat_min) & 
                              (bathy.lon < lon_max) ,
                              drop=True)
    
    return bathy_cut

def perpendicular(bathy, data_lat, data_lon, rot = np.pi/2):
    data_lat = - 22.9
    data_lon = -43.17

    ds = bathy.where((bathy.lat < data_lat +  0.5) & 
                              (bathy.lon < data_lon + 0.5) &
                              (bathy.lon > data_lon - 0.5) &
                              (bathy.lat > data_lat - 0.5) & 
                              (bathy.elevation  == 0),
                              drop=True)
    
    indices_zero = np.where(ds['elevation'].values == 0)

    # Pega os pontos mais extremos
    # talvez essa parte aqui de p melhorar
    latitude_sul, longitude_oeste = ds['lat'].values[indices_zero[0].min()], ds['lon'].values[indices_zero[1].min()]
    latitude_norte, longitude_leste = ds['lat'].values[indices_zero[0].max()], ds['lon'].values[indices_zero[1].max()]


    inclinação_linha = np.arctan2(latitude_norte - latitude_sul, longitude_leste - longitude_oeste)


    inclinação_normal = inclinação_linha  + rot

    # Calcular os pontos ao longo da linha normal
    distancia_normal = np.linspace(0, -2.5, 6)  # Ajuste conforme necessário

    lat_normal = data_lat + distancia_normal * np.sin(inclinação_normal)
    lon_normal = data_lon + distancia_normal * np.cos(inclinação_normal)

    return lat_normal, lon_normal

def plot_transec(bathy, lon_normal, lat_normal, data_lon, data_lat, fig_folder):
    bathy_lon, bathy_lat, bathy_h = bathy.lon, bathy.lat, bathy.elevation
    bathy_h = np.where(bathy_h > 0, np.nan, bathy_h)

    bathy_conts = np.array([-4000, -3000, -2000, -1000, -200, -100, -50, 0])


    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)
    bathy = ax.contourf(bathy_lon, bathy_lat, bathy_h, bathy_conts, transform=coord, cmap="Blues")
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    plt.plot(data_lon, data_lat, color='red', transform=ccrs.PlateCarree(), marker = 'o', label = 'Ponto Dado', linestyle=' ')

    ####  plota os pontos do transect    

    # for i,j, r in zip(lon_normal, lat_normal, range(len(lat_normal))):
    #     if r == 0:
    #         continue
    #     plt.plot(i, j, color='green', transform=ccrs.PlateCarree(), marker = 'o', label = 'Ponto', linestyle=' ')
    
    plt.plot(lon_normal, lat_normal, label='Linha normal à costa', color='white')

    #### essa parte comentada abaixo faz a linha nˆ
    ###  pra pegar o lon f, usa o gerar_pontos_linha logo abaixo

    # lon_medio = (lon_normal[-1] + lon_normal[0])/2
    # lat_medio = (lat_normal[-1] + lat_normal[0])/2
    # lon_f = -41.70827666
    # lat_f = -23.66410917

    # plt.plot([lon_medio, lon_f], [lat_medio, lat_f], label='Linha normal à costa', color='red')

    plt.legend(loc='lower left')
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
    lons = np.arange(bathy_lon.min().round(0), bathy_lon.max().round(0), 2)
    lats = np.arange(bathy_lat.min().round(0), bathy_lat.max().round(0), 2)
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
    plt.savefig(fig_folder + 'transec.png')

##### crazy stuff
    
def gerar_pontos_linha(inclinacao, ponto_inicial, comprimento):
    x0, y0 = ponto_inicial
    x = np.linspace(x0, x0 + comprimento, 2)  # Gera 100 pontos ao longo da linha
    y = inclinacao * (x - x0) + y0  # Equação da linha
    return x[-1], y[-1]

# inclinacao = inc_90 # no arquivo do_analysis
# ponto_inicial = (lon_medio, lat_medio) # na funcao de plotar
# comprimento = 1
# pf = gerar_pontos_linha(inclinacao, ponto_inicial, comprimento)

########## Old Stuff

def ponto_na_costa(ds, lat, lon):
    # Determinar os índices dos pontos vizinhos
    lat_idx = np.searchsorted(ds['lat'].values, lat)
    lon_idx = np.searchsorted(ds['lon'].values, lon)

    # Coordenadas dos pontos vizinhos
    # lat_vizinhos = ds['lat'].values[lat_idx-1:lat_idx+2]
    # lon_vizinhos = ds['lon'].values[lon_idx-1:lon_idx+2]

    # Verificar se há algum ponto adjacente com altitude inferior a 0
    # pontos_na_costa = ds['elevation'].sel(lat=lat_vizinhos, lon=lon_vizinhos).values < 0
    pontos_na_costa = ds['elevation'].sel(lat=slice(lat_idx-1, lat_idx+2),
                                           lon=slice(lon_idx-1, lon_idx+2)).values < 0


    if pontos_na_costa.all():
        return False
    else:
        return np.any(pontos_na_costa)

def plot_old_transec():
# criando o plot:
    coord = ccrs.PlateCarree()
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection=coord)
    # ax.set_extent([-42, -23, -60, -50], crs=coord)



    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)
    bathy = ax.contourf(bathy_lon, bathy_lat, bathy_h, bathy_conts, transform=coord, cmap="Blues")
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    # ax.set_extent([-40, 10, -50, -30], crs=coord)
    # plt.plot(if_lon1, if_lat1, color='green', transform=ccrs.PlateCarree(), marker = 'o')
    plt.plot(data_lon, data_lat, color='red', transform=ccrs.PlateCarree(), marker = 'o', label = 'Ilha Fiscal', linestyle=' ')
    plt.plot([data_lon, -41.1], [data_lat, -25.9], transform=ccrs.PlateCarree(), linestyle='-', color='red', linewidth=.8)
    # plt.plot(-42.2,-24.8, color = 'green', marker = 'o')
    plt.tight_layout()
    plt.legend(loc='lower left')
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
    lons = np.arange(bathy_lon.min().round(0), bathy_lon.max().round(0), 2)
    lats = np.arange(bathy_lat.min().round(0), bathy_lat.max().round(0), 2)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.savefig(fig_folder + 'faixa_if.png')
