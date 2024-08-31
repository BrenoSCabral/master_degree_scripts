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


bathy_file_path = Path('/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc')
bathy_ds_raw = xr.open_dataset(bathy_file_path)
# ds = bathy_ds_raw.where((bathy_ds_raw.lat < 5.5) & 
#                               (bathy_ds_raw.lon > -58) &
#                               (bathy_ds_raw.lat > -35) & 
#                               (bathy_ds_raw.lon < -28) ,
#                               drop=True)

ds = bathy_ds_raw.where((bathy_ds_raw.lat < -20) & 
                              (bathy_ds_raw.lon > -52) &
                              (bathy_ds_raw.lat > -30) & 
                              (bathy_ds_raw.lon < -40) ,
                              drop=True)



#####





# Extrair as variáveis relevantes: lat, lon e a batimetria (profundidade)
lat = ds['lat'].values
lon = ds['lon'].values
bathymetry = ds['elevation'].values  # Ou o nome da variável correspondente no seu arquivo
bathymetry = np.where(bathymetry > 0, np.nan, bathymetry)
# bathy50 = np.where(bathymetry == 50, bathymetry, np.nan)
bathy_conts = np.array([-3000, -2000, -1000, -200, -100, -50, 0])


# Ponto inicial (latitude e longitude)
initial_lat = -23.5
initial_lon = -43.09
lat_i_idx = np.abs(lat - initial_lat).argmin()
lon_i_idx = np.abs(lon - initial_lon).argmin()


lat_50, lon_50 = np.where(bathymetry == -50)

## new try
from scipy import spatial
cords_50 = np.array((lat_50, lon_50)).T # transformando agora em latlon
lat_idx, lon_idx = cords_50[spatial.KDTree(cords_50).query([initial_lat, initial_lon])[1]]

##### /new try

# id_lon2000 = np.abs(lon_2000 - lon_i_idx).argmin()

# lat_idx  = lat_2000[id_lat2000]
# lon_idx = lon_2000[id_lon2000]

post_lat = lat[lat_idx]
post_lon = lon[lon_idx]


# Calcular o gradiente da batimetria (usando diferenças finitas)
gradient_y, gradient_x = np.gradient(bathymetry)

# Determinar a direção do gradiente no ponto inicial
grad_direction = np.array([gradient_x[lat_idx, lon_idx], gradient_y[lat_idx, lon_idx]])
grad_direction /= np.linalg.norm(grad_direction)  # Normalizar o vetor

# Determinar a profundidade inicial
initial_depth = bathymetry[lat_idx, lon_idx]

# Parâmetros para a linha
target_depth = -3000  # profundidade desejada
n_points = 10  # número de pontos na linha

# Interpolar a linha ao longo do gradiente até atingir a profundidade alvo
depths = np.linspace(initial_depth, target_depth, n_points)
points = np.zeros((n_points, 2))  # array para armazenar [lat, lon] dos pontos

for i in range(n_points):
    # Atualizar a posição em função do gradiente
    points[i, 0] = initial_lat + grad_direction[1] * i * .3  # Ajuste o passo conforme necessário
    points[i, 1] = initial_lon + grad_direction[0] * i * .3



def pb():
    ####
    # criando o plot:
    coord = ccrs.PlateCarree()
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(111, projection=coord)
    # ax.set_extent([-42, -23, -60, -50], crs=coord)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)
    bathy = ax.contourf(lon, lat, bathymetry, bathy_conts, transform=coord, cmap="Blues")


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    plt.plot(initial_lon, initial_lat, 'o', color = 'purple')
    plt.plot(post_lon, post_lat, 'o', color = 'green')
    plt.plot(pts_2000[:,1], pts_2000[:,0],'r-o')
    # plt.plot(points[:, 1], points[:, 0], 'r-o')  # linha em vermelho com pontos marcados

    plt.tight_layout()
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
    plt.show()


pts_2000 = []
for i in cords_2000:
    pts_2000.append(np.array([lat[i[0]], lon[i[1]]]))
pts_2000 = np.array(pts_2000)