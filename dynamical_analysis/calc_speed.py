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


# my files
from read_reanalisys import set_reanalisys_dims
import filtro
sys.path.append(
    'old'
)
sys.path.append(
    'dynamical_analysis'
)
import plot_hovmoller as ph
import stats
import general_plots as gplots

import matplotlib
# matplotlib.use('TkAgg')



model_path = '/data3/MOVAR/modelos/REANALISES/'
# model_path = '/Users/breno/model_data/'
fig_folder = '/home/bcabral/mestrado/fig/isobaths_50/'


def get_reanalisys(lat, lon, model, di, df):
    reanal = {}
    years = list(set([di.year, df.year]))
    for year in years:
        # reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/*.nc')
        #                                   , model)        
        reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
                                             , model)
        
    reanalisys = xr.concat(list(reanal.values()), dim="time")
    model_series = reanalisys.sel(latitude=lat, longitude=lon, method='nearest')
    model_series = model_series.sel(time=slice(di, df))

    mod_ssh = model_series['ssh'].values
    # nao tava dando problema entao n ha necessidade de fazer assim
    # mod_ssh_to_filt = np.concatenate((np.full(len(mod_ssh)//2, mod_ssh.mean()), mod_ssh))
    # mod_time = pd.date_range(end=model_series['time'].values[-1], periods=len(mod_ssh_to_filt), freq='D')
    mod_time = model_series['time'].values
    mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)

    return reanalisys, mod_ssh, mod_band, mod_time


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


# def garbage():
#     # # Função para plotar o diagrama de Hovmöller
#     # def plot_hovmoller(hovmoller_data):
#     #     # intervals = np.array([-20, -10, 0, 10, 20])
#     #     plt.figure(figsize=(14, 8))
#     #     plt.contour(hovmoller_data.columns, hovmoller_data.index, hovmoller_data.values,
#     #                 levels=[-10, 10], colors='black', linestyles='-', linewidth = .05)
#     #     plt.contourf(hovmoller_data.columns, hovmoller_data.index, hovmoller_data.values, cmap='bwr')

#     #     plt.colorbar(label='SSH (cm)')
#     #     plt.xlabel('Mês')
#     #     plt.ylabel('Latitude')
#     #     plt.title('Diagrama de Hovmöller')
        
#     #     # Configurar o formato do eixo x para datas
#     #     plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#     #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     #     plt.gcf().autofmt_xdate()
        
#     #     plt.savefig('/Users/breno/mestrado/hovmoller.png')

#     # # Plotar o diagrama de Hovmöller

#     # def plot_hovmoller_continuous(hovmoller_data):
#     #     plt.figure(figsize=(14, 8))
        
#     #     # Define os níveis de SSH para o contorno contínuo
#     #     levels = np.linspace(-45, 45, 100)
        
#     #     # Plotar o preenchimento de contorno contínuo
#     #     c = plt.contourf(hovmoller_data.columns, hovmoller_data.index, hovmoller_data.values, levels=levels, cmap='bwr')
#     #     cbar = plt.colorbar(c)
#     #     cbar.set_label('SSH (cm)')
#     #     cbar.set_ticks([-20, -10, 0, 10, 20])  # Definir ticks específicos
#     #     cbar.ax.tick_params(labelsize=12) 
        
#     #     # Adicionar contornos preto para valores extremos
#     #     plt.contour(hovmoller_data.columns, hovmoller_data.index, hovmoller_data.values, levels=[-10, 10], colors='black', linestyles='-', linewidths=.5)
        
#     #     plt.xlabel('Tempo')
#     #     plt.ylabel('Latitude')
#     #     plt.title('Diagrama de Hovmöller com Contornos Contínuos')

#     #     # Configurar o formato do eixo x para datas
#     #     plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
#     #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#     #     plt.gcf().autofmt_xdate()
        
#     #     plt.savefig('/Users/breno/mestrado/hovmoller_sub20.png')
#     return 0

 
# Preparar os dados para o diagrama de Hovmöller

# # testar depois no ano de 2013 por ter um el nino fraco
# # link pra consulta -> https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php


di = datetime.datetime(2015,1,1)
df = datetime.datetime(2015,12,31)
# model = 'BRAN'
pts = get_points ()
models =  ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']

for model in models:
    df_ssh = collect_ssh_data(pts, di, df, model)

    # Hovmoller:
    hovmoller_data = ph.prepare_hovmoller_data(df_ssh) * 100 # passando pra m
    ph.plot_hovmoller(hovmoller_data, model=model)
    ph.plot_hovmoller_u20(hovmoller_data[hovmoller_data.index < -20], model=model)
    ph.plot_hovmoller_o20(hovmoller_data[hovmoller_data.index >= -20], model=model)


'''
TODO:
1 - Fazer o calculo da velocidade utilizando a media
'''

##########################
# calculando a velocidade --> Acho que não vai dar certo pela limitação física do meu camarada
##########################

#///// TENTATIVA 1 -> LAG DE CORRELACAO

import numpy as np
from scipy.signal import correlate
from scipy import signal
from geopy.distance import geodesic

def calculate_wave_speed(ssh1, ssh2, lat1, lon1, lat2, lon2):
    correlation = correlate(ssh1, ssh2, mode='full')
    lags = signal.correlation_lags(len(ssh1), len(ssh2), mode="full")
    lag = lags[np.argmax(abs(correlation))]
    print(f'lag = {lag}')

    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    distance = geodesic(point1, point2).kilometers


    speed = distance / lag
    
    return speed, lag, correlation

ssh1 = hovmoller_data.values[0]
ssh2 = hovmoller_data.values[2]
print(np.corrcoef(ssh1, ssh2)[1,0])
lat1, lon1 = pts.loc[0]['lat'], pts.loc[0]['lon']
lat2, lon2 = pts.loc[1]['lat'], pts.loc[1]['lon']


for i in range(len(hovmoller_data) - 4 ):
    if i == 0 or i==1 or i==2 or i==3:
        continue
    ssh1 = hovmoller_data.values[i-4]
    ssh2 = hovmoller_data.values[i+4]
    print(np.corrcoef(ssh1, ssh2)[1,0])
    lat1, lon1 = pts.loc[i-1]['lat'], pts.loc[i-1]['lon']
    lat2, lon2 = pts.loc[i+1]['lat'], pts.loc[i+1]['lon']
    velocidade, lag, correlacao = calculate_wave_speed(ssh1, ssh2, lat1, lon1, lat2, lon2)
    print(f"Lat0 = {lat1}, Latf = {lat2}\nVelocidade da onda: {velocidade/86.4} m/s\n______________")


from scipy import signal
ssh1 = hovmoller_data.values[0]
ssh2 = hovmoller_data.values[3]
correlation = signal.correlate(ssh1, ssh2 , mode="full")
lags = signal.correlation_lags(len(ssh1), len(ssh2), mode="full")
lag = lags[np.argmax(abs(correlation))]
lag


for i in range(len(hovmoller_data)):
    ssh1 = hovmoller_data.values[0]
    ssh2 = hovmoller_data.values[i]
    print(i)
    print(np.corrcoef(ssh1, ssh2)[1,0])
    print('_______')


ssh1 = hovmoller_data.values[0]
ssh2 = hovmoller_data.values[3]
print(np.corrcoef(ssh1[:-2], ssh2[2:])[1,0])

# ///// TENTATIVA 2 -> IDENTIFICAR PICO
import numpy as np
import pandas as pd

# Função para identificar picos ou fases em uma série temporal
def identify_peaks(data, threshold=0.5):
    peaks = []
    for i in range(1, len(data)-1):
        if (data[i] > data[i-1]) and (data[i] > data[i+1]) and (data[i] > threshold):
            peaks.append(i)
    return peaks

# Função para calcular a velocidade da onda entre duas latitudes
def calculate_wave_speed(hovmoller_data, lat1, lat2, threshold=0.5):
    # Extrair as séries temporais de SSH para as duas latitudes
    ssh1 = hovmoller_data.loc[lat1].values
    ssh2 = hovmoller_data.loc[lat2].values
    
    # Identificar os picos nas séries temporais
    peaks1 = identify_peaks(ssh1, threshold)
    peaks2 = identify_peaks(ssh2, threshold)
    
    # Se houver picos em ambas as latitudes
    if len(peaks1) > 0 and len(peaks2) > 0:
        # Calcular a diferença de tempo entre os picos
        time_diff = np.mean(np.diff(peaks2)) - np.mean(np.diff(peaks1))
        
        # Calcular a diferença de latitude
        lat_diff = lat2 - lat1
        
        # Calcular a velocidade
        speed = lat_diff / time_diff
        return speed
    else:
        return None

# Lista de latitudes no Hovmöller Data
latitudes = hovmoller_data.index

# Calculando as velocidades entre as latitudes consecutivas
velocidades = []

for i in range(len(latitudes) - 1):
    v = calculate_wave_speed(hovmoller_data, latitudes[i], latitudes[i+1])
    if v is not None:
        velocidades.append(v)

# Resultados
print("Variação da Velocidade das Ondas conforme a Latitude:", velocidades)
