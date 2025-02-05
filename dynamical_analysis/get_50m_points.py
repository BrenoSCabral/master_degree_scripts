# common libs
import os
import xarray as xr
import numpy as np
import json
import sys
import pandas as pd
import datetime
# from matplotlib import pyplot as plt

import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

# my files


import matplotlib
matplotlib.use('TkAgg')


model_path = '/data3/MOVAR/modelos/REANALISES/'
path_50m = '/home/bcabral/mestrado/lon_lat_50m.dat'
path_50m = '/Users/breno/mestrado/lon_lat_50m.dat'
# data_path =  f'/home/bcabral/mestrado/data/'
fig_folder = '/home/bcabral/mestrado/fig/isobaths_50/'



# def get_reanalisys(lat, lon, model, di, df):
#     reanal = {}
#     years = list(set([di.year, df.year]))
#     for year in years:
#         reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
#                                             , model)
        
#     reanalisys = xr.concat(list(reanal.values()), dim="time")
#     model_series = reanalisys.sel(latitude=lat, longitude=lon, method='nearest')
#     model_series = model_series.sel(time=slice(di, df))

#     mod_ssh = model_series['ssh'].values
#     # nao tava dando problema entao n ha necessidade de fazer assim
#     # mod_ssh_to_filt = np.concatenate((np.full(len(mod_ssh)//2, mod_ssh.mean()), mod_ssh))
#     # mod_time = pd.date_range(end=model_series['time'].values[-1], periods=len(mod_ssh_to_filt), freq='D')
#     mod_time = model_series['time'].values
#     mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)

#     return reanalisys, mod_ssh, mod_band, mod_time


pts = pd.read_csv(path_50m, header= None, sep = "\s+", names = ['lon', 'lat'])
pts = pts[pts['lon'] < 0].sample(frac=1).reset_index(drop=True) # tirando valores espurios
# pts = pts[~((pts['lon'] > -40) & (pts['lat'] < -20))].sample(frac=1).reset_index(drop=True)
# pts = pts[~((pts['lon'] > -40) & (pts['lat'] < -20))].sample(frac=1).reset_index(drop=True)
# vou pegar 1 ponto por grau


# pts = pts[pts['lat'] < -20].sample(frac=1).reset_index(drop=True) # tirando valores espurios


# # pegando de exemplo pra testar:
# lat = pts.iloc[0]['lat']
# lon = pts.iloc[0]['lon']

# # selecionando o ano de 2013 por ter um el nino fraco
# # link pra consulta -> https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
# di = datetime.datetime(2013,1,1)
# df = datetime.datetime(2013,12,31)
# rea_brute, reanalisys, fil_reanalisys, filt_time = get_reanalisys(lat, lon, 'BRAN', di, df)

# vou supor que vou usar os seguintes pontos:
# pts = pts[::200]
# aqui eu preciso pegar a reanalise nesses pontos e trackear um pico especifico, pra acompanhar
## o deslocamento e pegar assim a velocidade.

## plotar pontos da isobata de 50 m
# criando o plot:
# TODO: selecionar quais pontos pegar e plota-los
bathy_file_path = '/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc'
bathy_ds_raw = xr.open_dataset(bathy_file_path)
# ds = bathy_ds_raw.where((bathy_ds_raw.lat < 5.5) & 
#                               (bathy_ds_raw.lon > -58) &
#                               (bathy_ds_raw.lat > -35) & 
#                               (bathy_ds_raw.lon < -28) ,
#                               drop=True)

ds = bathy_ds_raw.where((bathy_ds_raw.lat > -40) ,
                              drop=True)

# Extrair as variáveis relevantes: lat, lon e a batimetria (profundidade)

lat = ds['lat'].values
lon = ds['lon'].values
bathymetry = ds['elevation'].values  # Ou o nome da variável correspondente no seu arquivo
bathymetry = np.where(bathymetry > 0, np.nan, bathymetry)
# bathymetry = np.where(bathymetry == -50, bathymetry, np.nan)
bathy_conts = np.array([-3000, -200, -50, 0])


pts_plot = pts[pts['lat'].between(-30.5, -30.4)]
pts_plot = pts
plot_pts(pts_plot)
# _____
# lat : lon
# - 33 : -51.74
# -32.5 : -51.44
# -32 : -51.21
# -31.5 :-50.84
# -31: -50.42
# - 30.5: 

'''
          lon        lat
-  -51.740000 -33.000000
-  -51.210000 -32.000000
-  -50.420000 -31.000000
0  -50.008333 -30.521528
1  -49.675000 -30.003125
2  -49.056746 -29.008333
3  -48.460366 -28.008333
4  -48.246930 -27.008333
5  -47.875000 -26.008333
6  -46.825000 -25.003030
7  -45.375000 -24.000177
8  -42.541667 -23.000088
9  -40.425000 -22.001190
10 -40.281667 -21.008333
11 -39.862634 -20.008333
12 -38.908333 -19.004932
13 -37.358333 -18.008333
14 -38.690000 -17.007180 ****** ALTERADO, era-37.525000
15 -37.975000 -16.006764
16 -38.843861 -15.008333
17 -38.850855 -14.008333
18 -38.441667 -13.002109
19 -37.477888 -12.008333
20 -36.918087 -11.008333
21 -35.753296 -10.008333
22 -34.886600  -9.008333
23 -34.550997  -8.008333
24 -34.555217  -7.008333
25 -34.958333  -6.002434
26 -35.099074  -5.008333
27 -37.512975  -4.008333
28 -38.877564  -3.008333
29 -42.870000  -2.000000 ****** ALTERADO, era -49.291667
30 -43.604459  -1.008333 
31 -44.641667  -0.002778
32 -46.941667   1.002976
33 -48.375000   2.000877
34 -49.508333   3.005348

### TEM QUE ALTERAR O DICT ABAIXO!!!

{'lon': {
  0: -51.74,
  1: -51.21,
  2: -50.42,
  3: -49.675,
  4: -49.056746,
  5: -48.460366,
  6: -48.24693,
  7: -47.875,
  8: -46.825,
  9: -45.375,
  10: -42.541667,
  11: -40.425,
  12: -40.281667,
  13: -39.862634,
  14: -38.908333,
  15: -37.358333,
  16: -38.690000,
  17: -37.975,
  18: -38.843861,
  19: -38.850855,
  20: -38.441667,
  21: -37.477888,
  22: -36.918087,
  23: -35.753296,
  24: -34.8866,
  25: -34.550997,
  26: -34.555217,
  27: -34.958333,
  28: -35.099074,
  29: -37.512975,
  30: -38.877564,
  31: -42.870000,
  32: -43.604459,
  33: -44.641667,
  34: -46.941667,
  35: -48.375,
  36: -49.508333},
 'lat': {0: -33.0,
  1: -32.0,
  2: -31.0,
  3: -30.003125,
  4: -29.008333,
  5: -28.008333,
  6: -27.008333,
  7: -26.008333,
  8: -25.00303,
  9: -24.000177,
  10: -23.000088,
  11: -22.00119,
  12: -21.008333,
  13: -20.008333,
  14: -19.004932,
  15: -18.008333,
  16: -17.00718,
  17: -16.006764,
  18: -15.008333,
  19: -14.008333,
  20: -13.002109,
  21: -12.008333,
  22: -11.008333,
  23: -10.008333,
  24: -9.0083333,
  25: -8.0083333,
  26: -7.0083333,
  27: -6.0024336,
  28: -5.0083333,
  29: -4.0083333,
  30: -3.0083333,
  31: -2.0009244,
  32: -1.0083333,
  33: -0.0027777778,
  34: 1.0029762,
  35: 2.0008772,
  36: 3.0053483}}
'''
pts_dict = {'lon': {
  0: -51.74,
  1: -51.21,
  2: -50.42,
  3: -49.675,
  4: -49.056746,
  5: -48.460366,
  6: -48.24693,
  7: -47.875,
  8: -46.825,
  9: -45.375,
  10: -42.541667,
  11: -40.425,
  12: -40.281667,
  13: -39.862634,
  14: -38.908333,
  15: -37.358333,
  16: -38.690000,
  17: -37.975,
  18: -38.843861,
  19: -38.850855,
  20: -38.441667,
  21: -37.477888,
  22: -36.918087,
  23: -35.753296,
  24: -34.8866,
  25: -34.550997,
  26: -34.555217,
  27: -34.958333,
  28: -35.099074,
  29: -37.512975,
  30: -38.877564,
  31: -42.870000,
  32: -43.604459,
  33: -44.641667,
  34: -46.941667,
  35: -48.375,
  36: -49.508333},
 'lat': {0: -33.0,
  1: -32.0,
  2: -31.0,
  3: -30.003125,
  4: -29.008333,
  5: -28.008333,
  6: -27.008333,
  7: -26.008333,
  8: -25.00303,
  9: -24.000177,
  10: -23.000088,
  11: -22.00119,
  12: -21.008333,
  13: -20.008333,
  14: -19.004932,
  15: -18.008333,
  16: -17.00718,
  17: -16.006764,
  18: -15.008333,
  19: -14.008333,
  20: -13.002109,
  21: -12.008333,
  22: -11.008333,
  23: -10.008333,
  24: -9.0083333,
  25: -8.0083333,
  26: -7.0083333,
  27: -6.0024336,
  28: -5.0083333,
  29: -4.0083333,
  30: -3.0083333,
  31: -2.0009244,
  32: -1.0083333,
  33: -0.0027777778,
  34: 1.0029762,
  35: 2.0008772,
  36: 3.0053483}}

pts_plot = pd.DataFrame(pts_dict)
bathy_conts = np.array([-3000, -200, -50, 0])

# plot_pts(pts_plot)

pts_wavelet = [-30, -20, -15, -10]

def plot_pts(pts_plot):
    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=coord)



    bathy = ax.contourf(lon, lat, bathymetry, bathy_conts, transform=coord, cmap="Blues")
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
    # fig = plt.figure(figsize=(20, 10))
    # ax.set_extent([-48, -29, -49, -31], crs=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    for index, row in pts_plot.iterrows():
        plt.plot(row['lon'], row['lat'],
                color='red', linewidth=2, marker='*',
                transform=ccrs.PlateCarree()
                )
        if round(row['lat'],1) in pts_wavelet:
          plt.plot(row['lon'], row['lat'],
                  color='yellow', linewidth=2, marker='o',
                  transform=ccrs.PlateCarree()
                  )

    # Ajustar os limites do mapa
    lat_min = pts_plot['lat'].min() - 1  # Ajuste conforme necessário
    lat_max = pts_plot['lat'].max() + 1  # Ajuste conforme necessário
    lon_min = pts_plot['lon'].min() - 1  # Ajuste conforme necessário
    lon_max = pts_plot['lon'].max() + 1  # Ajuste conforme necessário

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-50, -45, -40, -37, -35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.show()

    # plt.savefig('/Users/breno/mestrado/new_50m_pts.png')
    # plt.savefig(fig_folder + 'points', dpi=300)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.lines as mlines

def plot_pts_deepseek(pts_plot):
    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=coord)

    # Definindo as cores para cada intervalo
    colors = ["darkblue", "blue", "lightblue", "cyan"]
    bounds = [-3000, -200, -50, 0]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Usando o colormap personalizado
    bathy = ax.contourf(lon, lat, bathymetry, levels=bounds, transform=coord, cmap=cmap, norm=norm)
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)

    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    for index, row in pts_plot.iterrows():
        plt.plot(row['lon'], row['lat'],
                color='red', linewidth=2, marker='*',
                transform=ccrs.PlateCarree()
                )  

    lat_min = pts_plot['lat'].min() - 1
    lat_max = pts_plot['lat'].max() + 1
    lon_min = pts_plot['lon'].min() - 1
    lon_max = pts_plot['lon'].max() + 1

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-50, -45, -40, -37, -35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.show()


def plot_pts_deepseek2(pts_plot):
    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection=coord)

    # Extraindo cores específicas da paleta "Blues"
    blues_cmap = plt.get_cmap("Blues")  # Paleta Blues do Matplotlib
    colors = [
        blues_cmap(0.3),  # Tom mais escuro para -3000
        blues_cmap(0.5),  # Tom médio para -200
        blues_cmap(0.8),  # Tom mais claro para -50
        blues_cmap(1.0)   # Tom muito claro para 0
    ]

    # Definindo os intervalos
    bounds = [-3000, -200, -50, 0]
    cmap = mcolors.ListedColormap(colors)  # Criando um colormap com as cores extraídas
    norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalização para os intervalos definidos

    # Usando o colormap personalizado
    bathy = ax.contourf(lon, lat, bathymetry, levels=bounds, transform=coord, cmap=cmap, norm=norm)
    # fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)

    cbar = fig.colorbar(bathy, ax=ax, orientation="vertical", shrink=0.7, pad=0.08, aspect=40)
    cbar.set_label("Batimetria (m)", fontsize=14)  # Aumenta o tamanho da fonte do rótulo

    # Adicionando features geográficas
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    # Plotando os pontos
    for index, row in pts_plot.iterrows():
        if round(row['lat'],1) in pts_wavelet:
          plt.plot(row['lon'], row['lat'],
                  color='yellow', linewidth=4, marker='o',
                  transform=ccrs.PlateCarree(), markersize = 10
                  )
        plt.plot(row['lon'], row['lat'],
                color='red', linewidth=2, marker='*',
                transform=ccrs.PlateCarree()
                )

    # Ajustando os limites do mapa
    lat_min = pts_plot['lat'].min() - 1
    lat_max = pts_plot['lat'].max() + 1
    lon_min = pts_plot['lon'].min() - 1
    lon_max = pts_plot['lon'].max() + 1

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Adicionando gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-50, -45, -40, -35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
  # Criação das legendas manualmente

    red_legend = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                              markersize=10, label='Pontos utilizados para o Hovmöller')
    yellow_legend = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                                  markersize=10, label='Pontos utilizados para a Ondaleta')

    # Adicionando a legenda ao plot e posicionando fora do gráfico
    plt.legend(handles=[red_legend, yellow_legend], loc='upper center', 
              bbox_to_anchor=(0.5, -0.03), ncol=2, fontsize=14)

    # Ajustando o layout para evitar que a legenda seja cortada
    plt.tight_layout()


    # plt.show()


    plt.savefig('/Users/breno/mestrado/50m_pts_wav.png')

def plot_pts_deepseek_3(pts_plot):
  coord = ccrs.PlateCarree()
  fig = plt.figure(figsize=(12, 12))
  ax = fig.add_subplot(111, projection=coord)

  # Extraindo cores específicas da paleta "Blues"
  blues_cmap = plt.get_cmap("Blues")  # Paleta Blues do Matplotlib
  colors = [
      blues_cmap(0.3),  # Tom mais escuro para -3000
      blues_cmap(0.5),  # Tom médio para -200
      blues_cmap(0.8),  # Tom mais claro para -50
      blues_cmap(1.0)   # Tom muito claro para 0
  ]

  # Definindo os intervalos
  bounds = [-3000, -200, -50, 0]
  cmap = mcolors.ListedColormap(colors)  # Criando um colormap com as cores extraídas
  norm = mcolors.BoundaryNorm(bounds, cmap.N)  # Normalização para os intervalos definidos

  # Usando o colormap personalizado
  bathy = ax.contourf(lon, lat, bathymetry, levels=bounds, transform=coord, cmap=cmap, norm=norm)

  # Adicionando a colorbar
  cbar = fig.colorbar(bathy, ax=ax, orientation="vertical", shrink=0.7, pad=0.08, aspect=40)
  cbar.set_label("Batimetria (m)", fontsize=14)  # Aumenta o tamanho da fonte do rótulo

  # Adicionando features geográficas
  ax.add_feature(cfeature.LAND, facecolor='lightgray')
  ax.add_feature(cfeature.BORDERS, zorder=10)
  ax.add_feature(cfeature.COASTLINE, zorder=10)

  # Plotando os pontos
  for index, row in pts_plot.iterrows():
      if round(row['lat'], 1) in pts_wavelet:
          plt.plot(row['lon'], row['lat'],
                  color='yellow', linewidth=4, marker='o', markersize=10,
                  transform=ccrs.PlateCarree()
                  )
      plt.plot(row['lon'], row['lat'],
              color='red', linewidth=2, marker='*',
              transform=ccrs.PlateCarree()
              )

  # Ajustando os limites do mapa
  lat_min = pts_plot['lat'].min() - 1
  lat_max = pts_plot['lat'].max() + 1
  lon_min = pts_plot['lon'].min() - 1
  lon_max = pts_plot['lon'].max() + 1

  ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

  # Adicionando gridlines
  gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
  gl.xlabels_top = False
  gl.ylabels_left = False
  gl.xlocator = mticker.FixedLocator([-50, -45, -40, -35])
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER

  # Criação das legendas manualmente
  red_legend = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                            markersize=10, label='Pontos utilizados para o Hovmöller')
  yellow_legend = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                                markersize=10, label='Pontos utilizados para a Ondaleta')

  # Adicionando a legenda ao plot e posicionando abaixo do gráfico
  legend = plt.legend(handles=[red_legend, yellow_legend], loc='upper center',
                      bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=14, frameon=True)

  # Ajustando o layout para evitar que a legenda seja cortada
  plt.subplots_adjust(bottom=0.2)  # Ajusta o espaço abaixo do gráfico para acomodar a legenda

  # Salvando a figura
  plt.savefig('/Users/breno/mestrado/50m_pts_wav_t.png', bbox_inches='tight', dpi=300)
