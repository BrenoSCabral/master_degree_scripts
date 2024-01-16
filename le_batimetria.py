'''Rotina para ler batimetria da costa BR
base = https://notebook.community/ueapy/ueapy.github.io/content/notebooks/2019-05-30-cartopy-map
'''
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

# Open prepared bathymetry dataset using pathlib to sepcify the relative path


bathy_file_path = Path('/Users/breno/dados_mestrado/batimetria/gebco_bat.nc')
bathy_ds_raw = xr.open_dataset(bathy_file_path)
bathy_ds = bathy_ds_raw.where((bathy_ds_raw.lat < 5.5) & 
                              (bathy_ds_raw.lon > -58) &
                              (bathy_ds_raw.lat > -35) & 
                              (bathy_ds_raw.lon < -28) ,
                              drop=True)

bathy_lon, bathy_lat, bathy_h = bathy_ds.lon, bathy_ds.lat, bathy_ds.elevation
bathy_h = np.where(bathy_h > 0, np.nan, bathy_h)
bathy_conts = np.array([-3000, -2000, -1000, -200, -100, -50, 0])

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

plt.tight_layout()
fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)





pontos_goo = {
    'Cananeia' : [-25.02, -47.93],
    'Fernando de Noronha' : [-3.83, -32.40],
    'Fortaleza' : [-3.72, -38.47],
    'Ilha Fiscal' : [-22.90, -43.17],
    'Ilha Trindade' : [-20.50, -29.32],
    'Imbituba' : [-28.13, -48.40],
    'Macaé (Imbetiba)' : [-22.23, -41.47],
    'Rio Grande' : [-32.13, -52.10],
    'Salvador' : [-12.97, -38.52],
    'Santana' : [-0.06, -51.17],
    'São Pedro e São Paulo' : [3.83, -32.40],
    'Ubatuba' : [-23.50, -45.12]
}

pontos_sim = {
    'Rio Grande 2' : [-32.17, -52.09], #RS
    'Tramandaí' : [-30.00, -50.13], #RS
    'Paranaguá' : [-25.50, -48.53], #PR
    'Pontal do Sul' : [-25.55, -48.37], #PR
    'Ilhabela' : [-23.77, -45.35], #SP
    'DHN' : [-22.88, -43.13],#RJ
    'Ribamar': [-2.56, -44.05]#MA
    }

def convert(tude):
    multiplier = 1 if tude[-1] in ['N', 'E'] else -1
    return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(tude[:-1].split('-')))


fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection=coord)
bathy = ax.contourf(bathy_lon, bathy_lat, bathy_h, bathy_conts, transform=coord, cmap="Blues")


ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, zorder=10)
ax.add_feature(cfeature.COASTLINE, zorder=10)

# ax.set_extent([-40, 10, -50, -30], crs=coord)

for i in pontos_goo:
    plt.plot(pontos_goo[i][1], pontos_goo[i][0],
            color='red', linewidth=2, marker='P',
            transform=ccrs.PlateCarree()
            )  
    # plt.text(pontos_goo[i][1] - 0.05, pontos_goo[i][0] - 0.05, i,
    #         horizontalalignment='right', color = 'red', weight = 'bold',
    #         transform=ccrs.PlateCarree(), fontsize = 10)

for j in pontos_sim:
    plt.plot(pontos_sim[j][1], pontos_sim[j][0],
            color='green', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            )  
    # plt.text(pontos_sim[j][1] - 0.05, pontos_sim[j][0] - 0.05, j,
    #         horizontalalignment='right', color = 'green', weight = 'bold',
    #         transform=ccrs.PlateCarree(), fontsize = 10)

legend_elements = [
    Line2D([0],[0], color = 'red', marker='P', label='GOOS', markerfacecolor='red', markersize=15, linestyle = ''),
    Line2D([0],[0], color = 'green', marker='o', label='SiMCosta', markerfacecolor='green',
            markersize=15,  linestyle = '')
]

ax.legend(handles = legend_elements, loc='lower right')


plt.tight_layout()
fig.colorbar(bathy, ax=ax, orientation="vertical", label="(m)", shrink=0.7, pad=0.08, aspect=40)
# plt.savefig()