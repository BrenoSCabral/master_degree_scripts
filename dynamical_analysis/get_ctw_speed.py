import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from read_reanalisys import set_reanalisys_dims


'''
TODO:
- Abrir a batimetria
-- Selecionar a profundidade de 50 m - OK
-- Transformar essa profundidade em uma mascara - OK

- Abrir a reanalise
-- Selecionar somente os dados de 50 m usando a mascara da batimetria
-- calcular a velocidade


- Estou tendo dificuldade em aplicar a mascara dos 50 m na reanalise
vou tentar plotar o campo médio de ssh anual pra ter uma noção do que melhorar
'''

bathy_file_path = Path('/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc')
bathy_ds_raw = xr.open_dataset(bathy_file_path)
bathy_ds = bathy_ds_raw.where((bathy_ds_raw.lat < -20) & 
                              (bathy_ds_raw.lon > -52) &
                              (bathy_ds_raw.lat > -30) & 
                              (bathy_ds_raw.lon < -40) ,
                              drop=True)

bathy_ds = bathy_ds.rename({'lat': 'latitude', 'lon': 'longitude'})


depth_mask = bathy_ds['elevation'] == -50

depth_min = -60
depth_max = -40

# Aplicar a máscara para selecionar somente os dados entre -60 e -40 metros de profundidade
depth_band = bathy_ds.where((bathy_ds['elevation'] >= depth_min) & (bathy_ds['elevation'] <= depth_max), drop=True)

# Aplique a máscara para filtrar o dataset
bathy50 = bathy_ds.where(depth_mask, drop=False)

# abrindo arquivos TESTE

path_files = '/Users/breno/model_data/BRAN/'

model = set_reanalisys_dims(xr.open_mfdataset(path_files + '*.nc')
                                    , 'BRAN')

# Reamostrar batimetria para a grade do modelo
bathy_ds_regridded = bathy_ds['elevation'].interp(
    latitude=model['latitude'],
    longitude=model['longitude'],
    method='linear'
)


bathy_mask_expanded = bathy_mask.reindex_like(model_ds, method='nearest')
# Aplicar a máscara
model_masked = model_ds.where(bathy_mask_expanded, drop=True)

import xarray as xr
import numpy as np
from pathlib import Path

# Abrir o dataset de batimetria
bathy_file_path = Path('/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc')
bathy_ds_raw = xr.open_dataset(bathy_file_path)
bathy_ds = bathy_ds_raw.where(
    (bathy_ds_raw.lat < -20) & 
    (bathy_ds_raw.lon > -52) &
    (bathy_ds_raw.lat > -30) & 
    (bathy_ds_raw.lon < -40), 
    drop=True
)

# Criar a máscara para profundidade de 50 metros
depth_mask = bathy_ds['elevation'] == -50

# Reamostrar a batimetria para a grade do modelo
def regrid_bathy_to_model(bathy_ds, model_ds):
    # Reamostrar batimetria para a grade do modelo
    bathy_ds_regridded = bathy_ds['elevation'].interp(
        latitude=model_ds['latitude'],
        longitude=model_ds['longitude'],
        method='linear'
    )
    return bathy_ds_regridded

# Função para abrir e preparar o dataset de modelo
def prepare_model(path_files):
    # Abrir os arquivos do modelo
    model_ds = xr.open_mfdataset(path_files + '*.nc', combine='by_coords')
    # Supondo que você tenha uma função para ajustar as dimensões
    model_ds = set_reanalisys_dims(model_ds, 'BRAN')
    return model_ds

# Abrir e preparar o dataset de modelo
model_ds = prepare_model('/Users/breno/model_data/BRAN/')

# Reamostrar a batimetria para a grade do modelo
bathy_ds_regridded = regrid_bathy_to_model(bathy_ds, model_ds)

# Aplicar a máscara no modelo
def apply_mask_to_model(model_ds, bathy_mask):
    # Expandir a máscara para as dimensões do modelo
    bathy_mask_expanded = bathy_mask.reindex_like(model_ds, method='nearest')
    # Aplicar a máscara
    model_masked = model_ds.where(bathy_mask_expanded, drop=True)
    return model_masked

# Aplicar a máscara ao dataset de modelo
model_masked = apply_mask_to_model(model_ds, bathy_ds_regridded == -50)

# Verificar o resultado
print(model_masked)


## plot 

lon = bathy_ds_regridded['longitude']
lat = bathy_ds_regridded['latitude']
bathymetry = bathy_ds_regridded.values

ssh = model_masked.mean(dim= 'time').ssh.values
lon = model_masked['longitude']
lat = model_masked['latitude']


bathy_conts = np.array([-3000, -2000, -1000, -200, -100, -50, 0])
ssh_conts = np.array([0, 0.1, 0.2, .3,.4,.5,.6,.7,.8,.9,1])


bathymetry = depth_band.elevation.values
lon = depth_band['longitude']
lat = depth_band['latitude']

# criando o plot:
coord = ccrs.PlateCarree()

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(111, projection=coord)
# ax.set_extent([-42, -23, -60, -50], crs=coord)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=coord)
ax.set_extent([-40, -50, -20, -30], crs=ccrs.PlateCarree())
# bathy = ax.contourf(lon, lat, bathymetry, bathy_conts, transform=coord, cmap="Blues")
ssh_plot = ax.contourf(lon, lat, bathymetry, bathy_conts, transform=coord, cmap="Blues")


gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                    color='black', alpha=0.5, linestyle='--', draw_labels=True)

lons = model['longitude'].values
lats = model['latitude'].values
gl.xlocator = mticker.FixedLocator(lons)
gl.ylocator = mticker.FixedLocator(lats)

ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, zorder=10)
ax.add_feature(cfeature.COASTLINE, zorder=10)
# plt.plot(points[:, 1], points[:, 0], 'r-o')  # linha em vermelho com pontos marcados

# plt.tight_layout()
fig.colorbar(ssh_plot, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
plt.show()
