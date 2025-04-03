import pandas as pd
import xarray as xr
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.lines as mlines

bathy_file_path = '/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc'
bathy_ds_raw = xr.open_dataset(bathy_file_path)
# ds = bathy_ds_raw.where((bathy_ds_raw.lat < 5.5) & 
#                               (bathy_ds_raw.lon > -58) &
#                               (bathy_ds_raw.lat > -35) & 
#                               (bathy_ds_raw.lon < -28) ,
#                               drop=True)

ds = bathy_ds_raw.where((bathy_ds_raw.lat > -40) ,
                              drop=True)


lat = ds['lat'].values
lon = ds['lon'].values
bathymetry = ds['elevation'].values  # Ou o nome da variável correspondente no seu arquivo
bathymetry = np.where(bathymetry > 0, np.nan, bathymetry)
# bathymetry = np.where(bathymetry == -50, bathymetry, np.nan)
bathy_conts = np.array([-3000, -200, -50, 0])

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

pts_wavelet = [-30, -20, -15, -10]
########
## Pegando pontos de correte


def get_cross_points():
    # inicia em 0, nao em -50
    pts_cross = pd.DataFrame({
        'lon': {0: -50.57,
        1: -46.635435606453804,
        2: -47.85,
        3: -44.38592561317371,
        4: -42.02,
        5: -40.773547940652485,
        6: -40.81,
        7: -36.44319569143055,
        8: -38.99,
        9: -34.490344495807314,
        10: -37.05,
        11: -33.58410829011336,
        12: -36.83,
        13: -34.77723777841317}, 
        'lat': {0: -30.87,
        1: -33.05394323166,
        2: -25.0,
        3: -27.872384662368383,
        4: -23.0,
        5: -27.32397528552412,
        6: -21.0,
        7: -22.086933823530387,
        8: -15.0,
        9: -14.940794006523781,
        10: -11.0,
        11: -13.870191572249867,
        12: -5.0,
        13: -0.995431075920779}
    })
    return pts_cross


def interpolate_points(start, end, num_points):
    if num_points % 2 !=0:
        print('selecione um numero par de pontos!')
        num_points -= 1 
    lons = np.linspace(start[0], end[0], num_points)
    lats = np.linspace(start[1], end[1], num_points)
    return pd.DataFrame({'lon': lons, 'lat': lats})


pts_cross = get_cross_points()
sections = []
for i in range(len(pts_cross)):
    if i%2 != 0:
        continue
    start = (pts_cross.iloc[i]['lon'], pts_cross.iloc[i]['lat'])
    end = (pts_cross.iloc[i + 1]['lon'], pts_cross.iloc[i + 1]['lat'])

    cross_line = interpolate_points(start, end, 200)

    sections.append(cross_line)

secs = []
for section in sections:

    section = section[:-20] # corta uma parte de mar
    secs.append(section)


#######
#
#  PLOTANDO !!
#
###########

coord = ccrs.PlateCarree()
fig = plt.figure(figsize=(12,12))
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
cbar.set_label("Bathymetry (m)", fontsize=14)  # Aumenta o tamanho da fonte do rótulo



############

# 1. Banco de Abrolhos (polígono aproximado)


# Ajuste fino das coordenadas (valores em graus decimais)
# abrolhos_coords = [
#     (-37.7167, -17.9833),  # Arquipélago dos Abrolhos
#     (-38.3833, -18.5000),
#     (-38.7833, -19.0000),
#     (-38.2000, -19.2833),
#     (-37.2500, -18.7500),
#     (-37.7167, -17.9833)
# ]

abrolhos_coords = [
    (-39.81, -19.58),
    (-39.38, -19.71),
    (-38.66, -19.55),
    (-37.97, -19.28),
    (-37.72, -18.82),
    (-37.18, -18.21),
    (-37.53, -17.58),
    (-38.19, -17.22),
    (-39.14, -16.81),
    (-37.18, -18.21),
    (-37.53, -17.58),
    (-38.19, -17.22),
    (-39.14, -16.81),
    (-39.21, -17.76),
    (-40, -15),
    (-39.81, -19.58)
]

vitoria_trindade_coords = [
    (-40.3194, -20.3194),  # Vitória (ES)
    (-39.5000, -20.3000),  # Davis Bank
    (-38.0000, -20.5000),  # Jaseur Bank
    (-36.7833, -20.6167),  # Dogaressa Bank
    (-34.8333, -20.4167),  # Montague Bank
    (-29.3333, -20.5167)   # Ilha de Trindade
]

# ===========================================
# PLOT DAS FEIÇÕES MANUALMENTE
# ===========================================

# Banco de Abrolhos (área preenchida)
abrolhos_poly = plt.Polygon(
    abrolhos_coords,
    closed=True,
    edgecolor='lime',
    facecolor='lime',
    zorder=1,
    alpha=0.3,
    transform=ccrs.PlateCarree()
)
ax.add_patch(abrolhos_poly)

# Linha da Cadeia Vitória-Trindade
vt_line = plt.Line2D(
    [x[0] for x in vitoria_trindade_coords],
    [x[1] for x in vitoria_trindade_coords],
    color='darkorange',
    linewidth=2,
    zorder=2,
    linestyle='--',
    marker='o',
    markersize=5,
    transform=ccrs.PlateCarree()
)
ax.add_line(vt_line)

# ===========================================
# ANOTAÇÕES E LABELS
# ===========================================

# Label para Abrolhos
ax.text(
    -43, -18.5,
    'Abrolhos Bank',
    color='darkgreen',
    fontsize=12,
    zorder=5,
    ha='center',
    va='center',
    transform=ccrs.PlateCarree(),
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)

# Label para Cadeia Vitória-Trindade
ax.text(
    -50, -22,
    'Vitória-Trindade Ridge',
    color='darkorange',
    zorder=5,
    fontsize=12,
    rotation=10,
    transform=ccrs.PlateCarree(),
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)



############





# Adicionando features geográficas
ax.add_feature(cfeature.LAND, zorder=3, facecolor='lightgray')
ax.add_feature(cfeature.BORDERS, zorder=4)
ax.add_feature(cfeature.COASTLINE, zorder=4)

# plotando as secoes


# Adicionar linha conectando os pontos
for i, line in enumerate(secs, start=1):
    plt.plot(line['lon'], line['lat'], color='green', linestyle='--', transform=ccrs.PlateCarree())
    
    # Adicionar pontos nos extremos
    start = line.iloc[0]
    end = line.iloc[-1]
    plt.plot(start['lon'], start['lat'], color='green', linewidth=2, marker='o', transform=ccrs.PlateCarree())
    plt.plot(end['lon'], end['lat'], color='green', linewidth=2, marker='o', transform=ccrs.PlateCarree())
    
    # Adicionar número ao lado do ponto final
    plt.text(end['lon'] + 0.3, end['lat'], str(i), color='green', weight='bold', fontsize=12, transform=ccrs.PlateCarree())


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
                linewidth=2, color='gray', alpha=0.5, linestyle='--', zorder=4)
gl.xlabels_top = False
gl.ylabels_left = True
gl.ylabels_right = False
gl.xlocator = mticker.FixedLocator([-50, -45, -40, -27])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

# Criação das legendas manualmente
red_legend = mlines.Line2D([], [], color='red', marker='*', linestyle='None',
                        markersize=10, label='Hovmöller Points')
yellow_legend = mlines.Line2D([], [], color='yellow', marker='o', linestyle='None',
                            markersize=10, label='Wavelet Points')
green_legend = mlines.Line2D([], [], color='green', marker='o', linestyle='--',
                            markersize=10, label='Transections')

# Adicionando a legenda ao plot e posicionando abaixo do gráfico
legend = plt.legend(handles=[red_legend, yellow_legend, green_legend], loc='upper center',
                    bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12, frameon=True)

# Ajustando o layout para evitar que a legenda seja cortada
plt.subplots_adjust(bottom=0.2)  # Ajusta o espaço abaixo do gráfico para acomodar a legenda


# Salvando a figura
plt.savefig('/Users/breno/mestrado/points_analysis_features.png', dpi=300)
# plt.show()
