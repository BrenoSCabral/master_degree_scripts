import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os



def set_wind_reanalisys_dims(reanalisys, name):
    '''Changes the dimensions of xarray wind reanalisys objects to latitude, longitude, time, and v10

    Args:
        reanalisys (xarray Dataset): Dataset of the reanalisys data
        name (string): Name of the reanalisys

    Returns:
        xarray Dataset: Dataset of the reanalisys data with corrected dimensions names
    '''
    name = name.upper()
    if name == 'ERA-5':
        reanalisys = reanalisys.rename({'latitude': 'lat', 'longitude': 'lon'})
    elif name == 'MERRA2':
        reanalisys = reanalisys.rename({'V10M':'v10'})
    elif name == 'NCEP':
        reanalisys = reanalisys.rename({'vwnd':'v10'})

    return reanalisys


def plot_grid( reanalisys, nome_modelo):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

#     - Latitude: -52,5 a - 42,5
#    - Longitude: -70 a -60-

    ax.set_extent([-57.5, -72.5, -40, -55], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                        color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    lons = reanalisys['lon'].values
    lats = reanalisys['lat'].values
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    # ax.set_title(f'Ponto ')

    # plt.tight_layout()

    plt.title(nome_modelo + "'s grid")
    plt.tight_layout()

    # os.makedirs(f'/Users/breno/mestrado/CPAM/figs/data_mets/', exist_ok=True)
    plt.savefig(f'/Users/breno/mestrado/CPAM/figs/data_mets/points_{nome_modelo}.png')


def convert_lon(lon):
    return np.where(lon < 0, lon + 360, lon)


def convert_longitude_360_to_180(lon):
    return np.where(lon >= 180, lon - 360, lon)


def plot_mask(mask):
# Plotando o contorno preenchido
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 5))
    contour = ax.contourf(mask['lon'], mask['lat'], mask, cmap='viridis')

    # Adicionar linhas de contorno
    ax.contour(mask['lon'], mask['lat'], mask, colors='k', linewidths=0.5)

    # Adicionar feições cartográficas
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, zorder=0, edgecolor='black')

    # Adicionar barra de cores
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical', shrink=0.5, label='Máscara de Terra/Água')

    # Configurar limites do mapa
    ax.set_extent([-70, -60, -52.5, -42.5], crs=ccrs.PlateCarree())

    # Título do mapa
    ax.set_title('Máscara de Terra/Água', fontsize=15)

    plt.show()


models_path = '/Users/breno/mestrado/CPAM/models/'

models = ['ERA-5', 'MERRA2', 'NCEP']
# NEED TO rename every MERRA2 file to end with .nc

## import my reanalisys
era_path = models_path + '/ERA-5/'

reanalera = xr.open_mfdataset(era_path+'*.nc')
reanalmerra = xr.open_mfdataset(models_path + '/MERRA2/'+'*.nc')
reanalncep = xr.open_mfdataset(models_path + '/NCEP/'+'*.nc')

reanal = reanalera
model = 'ERA-5'

## set its dimensions names

reanal = set_wind_reanalisys_dims(reanal, model)

## Plot the grids of each model

plot_grid(reanal, model)


## select only ocean data

mask_path = f'{models_path}IMERG_land_sea_mask.nc'
ds_mask = xr.open_dataset(mask_path)
ocean_mask = ds_mask['landseamask']

lat_range = slice(-52.5, -42.5)
lon_range = slice(-70, -60) # in 0 to 360
lon_range_mask = slice(convert_lon(-70), convert_lon(-60)) # in 0 to 360

mask = ocean_mask.sel(lat=lat_range, lon=lon_range_mask)

lon_original = mask['lon'].values
lon_convertido = convert_longitude_360_to_180(lon_original)
mask = mask.assign_coords(lon=lon_convertido)

mask_interpolated = mask.interp(lat=reanal['v10']['lat'], lon=reanal['v10']['lon'])

v10_ocean = reanal['v10'].where(mask_interpolated == 100) # maximo de agua = 100

## pega a correlacao do espectro dele com a dos pontos

'''
TODO:

Agora eu vou precisar importar uma serie de dado (comecar importando ilha fiscal 2014, como de praxe)
No meu teste, que ta na pasta CPAM dos scripts, eu so reamostrei e comparei, mas acho que o ideal eh
filtrar pelo menos o ssh e comparar com o vento.

Provavelmente vou precisar fazer um loop pra pegar todos os pontos do modelo e ver qual possui maior corr. 
Interessante seria fazer um contourf que nem fiz no ponto dos SPAOs (so que agr meu contourf seria mto maior)

Uma vez que eu peguei o ponto de maior correlacao, uso ele pra fazer o crosspecs e vejo a coerencia no sinal.
Se der tempo, seria legal pegar a regiao de ciclogenese no RS tbm, pra ver qual acaba representando melhor.

No final isso vai dar um trabalho legal.
'''

