# Arquivo criado com a intencao de comparar o ssh com a corrente
model_path = '/data3/MOVAR/modelos/REANALISES/'
# model_path = '/Users/breno/model_data/BRAN_CURR'
model = 'BRAN'
fig_folder = '/home/bcabral/mestrado/fig/isobaths_50/'

import os
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd
import datetime


import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


sys.path.append(
    '../dynamical_analysis'
)
sys.path.append(
    '../'
)
# my files
from read_reanalisys import set_reanalisys_curr_dims, set_reanalisys_dims
import filtro
# import plot_hovmoller as ph
import model_filt
# import general_plots as gplots

# sys.path.append(
#     '../old'
# )
# import stats


def get_points():
    pts = {'lon': {
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
    return pd.DataFrame(pts)


def get_cross_points():
    pts_cross = pd.DataFrame({
        'lon':{0:-49.675,
            1: -47.52,
            2: -46.825,
            3: -44.63,
            4: -42.541667,
            5: -40.89,
            6: -40.281667,
            7: -38.61,
            8:-38.843861,
            9: -38.23,
            10:-36.918087,
            11: -36.44,
            12:-35.099074,
            13: -34.76},
        'lat':{0:-30.003125,
            1: - 31.05,
            2: - 25,
            3: -27.67,
            4: -23,
            5: -26.92,
            6: -21,
            7: -21.92,
            8: -15,
            9: -14.99,
            10:-11,
            11: -11.54,
            12:- 5,
            13: -4.9}
        }
    )
    return pts_cross


def collect_ssh_data(pts, di, df, model):
    ssh_data = []
    lats = []
    times = []
    
    for index, row in pts.iterrows():
        lat = row['lat']
        lon = row['lon']
        _, reanalisys, fil_reanalisys, filt_time = get_reanalisys(lat, lon, model, di, df)
        
        # Armazenar latitude, tempo e dados de SSH
        lats.extend([lat] * len(filt_time))
        times.extend(filt_time)
        ssh_data.extend(fil_reanalisys)
    
    # Criar DataFrame para os dados
    df_ssh = pd.DataFrame({
        'time': times,
        'lat': lats,
        'ssh': ssh_data
    })
    
    return df_ssh


def collect_curr_data(pts, di, df, model):
    ssh_data = []
    lats = []
    times = []
    
    for index, row in pts.iterrows():
        lat = row['lat']
        lon = row['lon']
        _, reanalisys, fil_reanalisys, filt_time = get_reanalisys(lat, lon, model, di, df)
        
        # Armazenar latitude, tempo e dados de SSH
        lats.extend([lat] * len(filt_time))
        times.extend(filt_time)
        ssh_data.extend(fil_reanalisys)
    
    # Criar DataFrame para os dados
    df_ssh = pd.DataFrame({
        'time': times,
        'lat': lats,
        'ssh': ssh_data
    })
    
    return df_ssh


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


def CalcGeographicAngle(arith):
    return (360 - arith + 90) % 360


# selecionar ponto
pts_cross = get_cross_points()
sections = []
for i in range(len(pts_cross)):
    if i%2 != 0:
        continue
    start = (pts_cross.iloc[i]['lon'], pts_cross.iloc[i]['lat'])
    end = (pts_cross.iloc[i + 1]['lon'], pts_cross.iloc[i + 1]['lat'])

    cross_line = interpolate_points(start, end, 10)

    sections.append(cross_line)

section = sections[0]
lat = section['lat'][0]
lon = section['lon'][0]

years = [2015]
# importar e filtrar dado de ssh
reanal = {}
for year in years:
    # reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/*.nc')
    #                                   , model)        
    reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                            , model)
    
reanalisys = xr.concat(list(reanal.values()), dim="time")
model_series = reanalisys.sel(latitude=lat, longitude=lon, method='nearest')

mod_ssh = model_series['ssh'].values
mod_time = model_series['time'].values
mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)


# importar e filtrar dado de curr
reanal_curr = {}

for year in years:
    reanal_curr[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                        , model)
    
reanalisys_curr = xr.concat(list(reanal_curr.values()), dim="time")
model_series_curr = reanalisys_curr.sel(latitude=lat, longitude=lon, depth=0, method='nearest')

u_filt = model_filt.filtra_reanalise_u(model_series_curr)
v_filt = model_filt.filtra_reanalise_v(model_series_curr)

# dependendo do modelo que for usar, acho que nao precisa nem interpolar. Se for necessario, achei essa resposta
# -> https://stackoverflow.com/questions/73455966/regrid-xarray-dataset-to-a-grid-that-is-2-5-times-coarser



delta_lon = section['lon'].values[-1] - section['lon'].values[0]
delta_lat = section['lat'].values[-1] - section['lat'].values[0]
theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
theta_deg = np.degrees(theta_rad)  # Convertendo para graus

# theta_geo = CalcGeographicAngle(theta_deg)
# theta_geo_rad = np.radians(theta_geo)  # Converter de volta para radianos

# ou compoe a intensidade e multiplica pelo cos do angulo
section_int = np.sqrt(u_filt **2 + v_filt**2)
section_int_rotated = section_int * np.cos(theta_rad)


section_n_filt_int = (np.sqrt(model_series_curr['u'] **2 + model_series_curr['v']**2)).values
section_n_filt_int_rotated = section_n_filt_int * np.cos(theta_rad)

# TODO: plotar em conjunto os dados de ssh e curr (cm no eixo x esquerdo, m/s no eixo y direito)
import matplotlib.pyplot as plt
import numpy as np
t = reanalisys.time.values


fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('ssh (cm)', color=color)
ax1.plot(t, mod_band*100, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('curr (m/s)', color=color)  # we already handled the x-label with ax1
ax2.plot(t, section_int_rotated, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'/home/bcabral/mestrado/fig/ssh_curr.png')

# TODO: plotar a serie historica de um ponto de corrente filtrado vs nao filtrado

fig, ax1 = plt.subplots(figsize=(10,6))

color = 'tab:red'
ax1.set_xlabel('time')
ax1.set_ylabel('curr (m/s)', color=color)
ax1.plot(t, section_n_filt_int_rotated, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('curr filt (m/s)', color=color)  # we already handled the x-label with ax1
ax2.plot(t, section_int_rotated, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'/home/bcabral/mestrado/fig/curr_nfilt_pt0.png')
