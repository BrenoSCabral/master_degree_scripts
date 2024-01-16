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

fig_folder = '/Users/breno/Documents/Mestrado/tese/figs/rep3/'
path_dado = '/Users/breno/dados_mestrado/dados/'
path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR4/'

# filename = 'Ilha Fiscal 2014.txt'
# formato = 'csv'
nome = 'Ilha Fiscal'

if_lat2 = -22.9
if_lon2 = -43.17

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




bathy_file_path = Path('/Users/breno/dados_mestrado/batimetria/gebco_bat.nc')
bathy_ds_raw = xr.open_dataset(bathy_file_path)
bathy_ds = bathy_ds_raw.where((bathy_ds_raw.lat < -19) & 
                              (bathy_ds_raw.lon > -48) &
                              (bathy_ds_raw.lat > -26) & 
                              (bathy_ds_raw.lon < -38) ,
                              drop=True)

bathy_lon, bathy_lat, bathy_h = bathy_ds.lon, bathy_ds.lat, bathy_ds.elevation
bathy_h = np.where(bathy_h > 0, np.nan, bathy_h)
bathy_conts = np.array([-4000, -3000, -2000, -1000, -200, -100, -50, 0])

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
plt.plot(if_lon2, if_lat2, color='red', transform=ccrs.PlateCarree(), marker = 'o', label = 'Ilha Fiscal', linestyle=' ')
plt.plot([if_lon2, -41.1], [if_lat2, -25.9], transform=ccrs.PlateCarree(), linestyle='-', color='red', linewidth=.8)
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
############ \\\\\\\\
