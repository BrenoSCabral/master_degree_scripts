''' 
>TODO<
1 - Plotar os pontos de 50 utilizados pro hovmoller. - OK
2 - Escolher os 7 pontos (30, 25, 23, 21, 15 ,11, 5) - OK
3 - adaptar a funcao pra cuspir o angulo ao inves de uma linha unitaria
3 - converter a angulacao do angulo normal a linha pra angulo N ao inves de trigonometrico
'''

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker

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


bathy_file_path = '/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc'
bathy_ds_raw = xr.open_dataset(bathy_file_path)
# ds = bathy_ds_raw.where((bathy_ds_raw.lat < 5.5) & 
#                               (bathy_ds_raw.lon > -58) &
#                               (bathy_ds_raw.lat > -35) & 
#                               (bathy_ds_raw.lon < -28) ,
#                               drop=True)

# pra agilizar os plots:
ds = bathy_ds_raw.where((bathy_ds_raw.lat > -40) ,
                              drop=True)

# Extrair as variáveis relevantes: lat, lon e a batimetria (profundidade)

lat = ds['lat'].values
lon = ds['lon'].values
bathymetry = ds['elevation'].values  # Ou o nome da variável correspondente no seu arquivo
bathymetry = np.where(bathymetry > 0, np.nan, bathymetry)
# bathymetry = np.where(bathymetry == -50, bathymetry, np.nan)
bathy_conts = np.array([-3000,-2500, -2000, -1500, -1000, -500, -100, -50, 0])

def plot_pts(pts_plot):
    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8))
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

    # Ajustar os limites do mapa
    # lat_min = pts_plot['lat'].min() - 1  # Ajuste conforme necessário
    # lat_max = pts_plot['lat'].max() + 1  # Ajuste conforme necessário
    # lon_min = pts_plot['lon'].min() - 1  # Ajuste conforme necessário
    # lon_max = pts_plot['lon'].max() + 1  # Ajuste conforme necessário

    # ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-50, -45, -40, -37, -35])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.show()

    # plt.savefig('/Users/breno/mestrado/pts.png', dpi = 300)
    # plt.savefig(fig_folder + 'points', dpi=300)


def plot_line(lines_plot):
    coord = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection=coord)



    bathy = ax.contourf(lon, lat, bathymetry, bathy_conts, transform=coord, cmap="Blues")
    fig.colorbar(bathy, ax=ax, orientation="vertical", label="Batimetria (m)", shrink=0.7, pad=0.08, aspect=40)
    # fig = plt.figure(figsize=(20, 10))
    # ax.set_extent([-48, -29, -49, -31], crs=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)


    # Adicionar linha conectando os pontos
    for line in lines_plot:
        plt.plot(line['lon'], line['lat'], color='red', linestyle='--', transform=ccrs.PlateCarree())
        
        # Adicionar pontos nos extremos
        start = line.iloc[0]
        end = line.iloc[-1]
        plt.plot(start['lon'], start['lat'], color='red', linewidth=2, marker='*', transform=ccrs.PlateCarree())
        plt.plot(end['lon'], end['lat'], color='red', linewidth=2, marker='*', transform=ccrs.PlateCarree())
        

        # point = line.iloc[len(line)//2]


        # lon_p, lat_p = point['lon'], point['lat']
        # angle = np.degrees(np.arctan2(end[1] - start[1], end[0] - start[0]))
        # perp_line = perpendicular_line(lon_p, lat_p, angle - 90, length=2)
        # plt.plot([perp_line[0][0], perp_line[1][0]], [perp_line[0][1], perp_line[1][1]], color='red', linestyle='--', transform=ccrs.PlateCarree())


    # # Ajustar os limites do mapa
    lat_min = -35  # Ajuste conforme necessário
    lat_max = 0 # Ajuste conforme necessário
    lon_min = -55  # Ajuste conforme necessário
    lon_max = -30 # Ajuste conforme necessário

    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())


    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlocator = mticker.FixedLocator([-55, -50, -45, -40, -35, -30])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.show()
    plt.savefig('/Users/breno/mestrado/linhas_longshore_adj.png')


def interpolate_points(start, end, num_points):
    if num_points % 2 !=0:
        print('selecione um numero par de pontos!')
        num_points -= 1 
    lons = np.linspace(start[0], end[0], num_points)
    lats = np.linspace(start[1], end[1], num_points)
    return pd.DataFrame({'lon': lons, 'lat': lats})


def perpendicular_line(lon, lat, angle, length=1):
    angle_rad = np.deg2rad(angle)
    dlon = length * np.cos(angle_rad)
    dlat = length * np.sin(angle_rad)
    return [(lon + dlon, lat + dlat), (lon, lat)]


# inicia em 0, nao em -50
pts_cross = pd.DataFrame({
    'lon':{0:-50.57,
        1: -48.3,
        2: -47.85,
        3: -44.63,
        4: -42.02,
        5: -40.89,
        6: -40.81,
        7: -38.52,
        8:-38.99,
        9: -38.23,
        10:-37.05,
        11: -36.41,
        12:-36.83,
        13: -36.22},
    'lat':{0:-30.87,
        1: - 32.13,
        2: - 25,
        3: -27.67,
        4: -23,
        5: -26.92,
        6: -21,
        7: -21.57,
        8: -15,
        9: -14.99,
        10:-11,
        11: -11.53,
        12:- 5,
        13: -3.81}
    }
)

# pts_cross = pd.DataFrame({
#     'lon':{0:-49.675,
#            1: -47.52,
#            2: -46.825,
#            3: -44.63,
#            4: -42.541667,
#            5: -40.89,
#            6: -40.281667,
#            7: -38.61,
#            8:-38.843861,
#            9: -38.23,
#            10:-36.918087,
#            11: -36.44,
#            12:-35.099074,
#            13: -34.76},
#     'lat':{0:-30.003125,
#            1: - 31.05,
#            2: - 25,
#            3: -27.67,
#            4: -23,
#            5: -26.92,
#            6: -21,
#            7: -21.92,
#            8: -15,
#            9: -14.99,
#            10:-11,
#            11: -11.54,
#            12:- 5,
#            13: -4.9}
#     }
# )

# plot_pts(pts_cross)

# Função para calcular o novo ponto final
def calculate_new_endpoints(df, distance_km=500, km_per_degree=111.11):
    new_points = []
    
    for i in range(0, len(df), 2):
        # Coordenadas iniciais
        lat1, lon1 = df.loc[i, 'lat'], df.loc[i, 'lon']
        lat2, lon2 = df.loc[i + 1, 'lat'], df.loc[i + 1, 'lon']
        
        # Diferenças de latitude e longitude
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        
        # Ângulo da linha
        angle = np.arctan2(delta_lat, delta_lon)
        
        # Deslocamentos para 300 km
        delta_lat_new = (distance_km * np.sin(angle)) / km_per_degree
        delta_lon_new = (distance_km * np.cos(angle)) / km_per_degree
        
        # Novo ponto final
        new_lat = lat1 + delta_lat_new
        new_lon = lon1 + delta_lon_new
        
        # Adicionar ponto inicial e novo final à lista
        new_points.append({'lat': lat1, 'lon': lon1})
        new_points.append({'lat': new_lat, 'lon': new_lon})
    
    return pd.DataFrame(new_points)


pts_cross = calculate_new_endpoints(pts_cross)



lines_plot = []
for i in range(len(pts_cross)):
    if i%2 != 0:
        continue
    start = (pts_cross.iloc[i]['lon'], pts_cross.iloc[i]['lat'])
    end = (pts_cross.iloc[i + 1]['lon'], pts_cross.iloc[i + 1]['lat'])

    cross_line = interpolate_points(start, end, 10)

    lines_plot.append(cross_line)
plot_line(secs)
print(pts_cross)