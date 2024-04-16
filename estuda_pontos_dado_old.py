# feito pro servidor do LOF

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import matplotlib.ticker as mticker
import os
# from le_reanalise import get_lat_lon, set_reanalisys_dims
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
# acho que vou pegar a resolucao de cada modelo e pegar uma malha de 9 pontos pra cada.

home_path =  '/home/bcabral/'

def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude'})

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
for model in modelos:
    path_model = f'/data3/MOVAR/modelos/REANALISES/{model}/SSH/2014/'
    reanalisys = xr.open_mfdataset(path_model + '*.nc')
    reanalisys = set_reanalisys_dims(reanalisys, model)
    print('____OK____' + model)

    # aqui pra baixo precisa ler o arquivo todo e so depois fazer o sel na latlon pra comparar