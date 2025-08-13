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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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


def subplot_map(ax, s):
    # Extraindo cores específicas da paleta "Blues"
    coord = ccrs.PlateCarree()

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
    # cbar = fig.colorbar(bathy, ax=ax, orientation="vertical", shrink=0.7, pad=0.08, aspect=40)
    # cbar.set_label("Bathymetry (m)", fontsize=14)  # Aumenta o tamanho da fonte do rótulo



    # Adicionando features geográficas
    ax.add_feature(cfeature.LAND, zorder=3, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=4)
    ax.add_feature(cfeature.COASTLINE, zorder=4)

    # plotando as secoes


    # Adicionar linha conectando os pontos
    for i, line in enumerate(secs, start=1):
        if i !=s:
            continue
        ax.plot(line['lon'], line['lat'], color='green', linestyle='--', transform=ccrs.PlateCarree())
        
        # Adicionar pontos nos extremos
        start = line.iloc[0]
        end = line.iloc[-1]
        ax.plot(start['lon'], start['lat'], color='green', linewidth=2, marker='o', transform=ccrs.PlateCarree())
        ax.plot(end['lon'], end['lat'], color='green', linewidth=2, marker='o', transform=ccrs.PlateCarree())
        
        # Adicionar número ao lado do ponto final
        ax.text(end['lon'] + 0.3, end['lat'], str(i), color='green', weight='bold', fontsize=12, transform=ccrs.PlateCarree())


    # Ajustando os limites do mapa
    lat_min = secs[s-1]['lat'].mean() -5
    lat_max = secs[s-1]['lat'].mean() +5
    lon_min = secs[s-1]['lon'].mean() -5
    lon_max = secs[s-1]['lon'].mean() +5

    # lat_min = pts_plot['lat'].min() - 1
    # lat_max = pts_plot['lat'].max() + 1
    # lon_min = pts_plot['lon'].min() - 1
    # lon_max = pts_plot['lon'].max() + 1

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Adicionando gridlines
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--', zorder=4)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.xlabels_bot = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([-56, -53, -50, -47,  -44,  -41,  -38, -35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    return ax

