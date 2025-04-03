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
sys.path.append(
    '../'
)
from read_reanalisys import set_reanalisys_dims
import filtro
import plot_hovmoller as ph
# import stats
# import general_plots as gplots

import matplotlib
matplotlib.use('TkAgg')



model_path = '/data3/MOVAR/modelos/REANALISES/'
# model_path = '/Users/breno/model_data/'
fig_folder = '/home/bcabral/mestrado/fig/hovemoller/'


def get_reanalisys(lat, lon, model, di, df):
    reanal = {}
    years = list(set([di.year, df.year]))
    for year in years:
        reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + str(year)  + '/*.nc')
                                           , model)        
        # reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
        #                                      , model)
        
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

pts = get_points ()
model = 'BRAN'

for year in range(1995, 2024):
    di = datetime.datetime(year,1,1)
    df = datetime.datetime(year,12,31)
    # model = 'BRAN'
    # models =  ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']

    # for model in models:
    df_ssh = collect_ssh_data(pts, di, df, model)

    # Hovmoller:
    hovmoller_data = ph.prepare_hovmoller_data(df_ssh) * 100 # passando pra m
    hovmoller_data.to_csv('/Users/breno/mestrado/hov_data.csv')
    print(f"comecou os plots de {year}")
    ph.plot_hovmoller(hovmoller_data, model=model, fig_folder=fig_folder)
    # ph.plot_hovmoller_u20(hovmoller_data[hovmoller_data.index < -20], model=model, fig_folder=fig_folder)
    #ph.plot_hovmoller_o20(hovmoller_data[hovmoller_data.index >= -20], model=model, fig_folder=fig_folder)


######
# tentando calcular coerencia
#####
from scipy.signal import coherence
pd.read_csv('/Users/breno/mestrado/hov_data.csv', index_col=0)

def plot_coherence(signal1, signal2, fs=1.0, title='Coerência'):
    f, Cxy = coherence(signal1, signal2, fs=fs, window='hann', nperseg=256)
    plt.figure()
    plt.plot(f, Cxy)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Coerência')
    plt.title(title)
    plt.show()

data = hovmoller_data.T.copy()

# Exemplo: Calcular a coerência entre duas latitudes
latitudes = data.columns  # Lista de latitudes
lat1, lat2 = -33.000000, -32.000000  # Escolha duas latitudes para comparar

# Extraia as séries temporais para as latitudes escolhidas
signal1 = data[lat1].values
signal2 = data[lat2].values

# Calcule e plote a coerência
plot_coherence(signal1, signal2, fs=1.0, title=f'Coerência entre {lat1} e {lat2}')


#######
# coerencia mapa
#####

from scipy.signal import coherence

# Lista de latitudes
latitudes = data.columns
# latitudes = latitudes[10:]

# Matriz para armazenar a coerência média entre pares
coherence_matrix = np.zeros((len(latitudes), len(latitudes)))

# Loop para calcular a coerência entre todos os pares
for i, lat1 in enumerate(latitudes):
    for j, lat2 in enumerate(latitudes):
        if i < j:  # Evitar cálculos redundantes
            signal1 = data[lat1].values
            signal2 = data[lat2].values
            f, Cxy = coherence(signal1, signal2, fs=1.0, window='hann', nperseg=256)
            coherence_matrix[i, j] = np.mean(Cxy)  # Coerência média

# Visualize a matriz de coerência
plt.imshow(coherence_matrix, cmap='viridis', origin='lower', extent=[latitudes.min(), latitudes.max(), latitudes.min(), latitudes.max()])
plt.colorbar(label='Coerência Média')
plt.xlabel('Latitude')
plt.ylabel('Latitude')
plt.title('Mapa de Coerência')
plt.show()


####
# com valores
####

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Gerar dados de exemplo (substitua pelo seu coherence_matrix)
coherence_matrix = np.random.rand(len(latitudes), len(latitudes))  # Exemplo aleatório
np.fill_diagonal(coherence_matrix, 1)  # Coerência de um sinal consigo mesmo é 1

# Plotar o mapa de calor
plt.figure(figsize=(10, 8))
sns.heatmap(coherence_matrix, annot=True, fmt=".1f", cmap="viridis",
            xticklabels=latitudes, yticklabels=latitudes,
            cbar_kws={'label': 'Coerência Média'})

plt.title('Mapa de Coerência com Valores Anotados', fontsize=16)
plt.xlabel('Latitude', fontsize=14)
plt.ylabel('Latitude', fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#####
# dois pts
####

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

# Função para calcular e plotar a coerência em função do período
def plot_coherence_period(signal1, signal2, fs=1.0, title='Coerência'):
    # Calcular a coerência e as frequências
    f, Cxy = coherence(signal1, signal2, fs=fs, window='hann', nperseg=256)
    
    # Converter frequência (Hz) para período (dias)
    period = 1 / f  # Período em dias (assumindo fs=1/dia)
    
    # Plotar a coerência em função do período
    plt.figure(figsize=(10, 6))
    plt.plot(period, Cxy, label='Coerência')
    plt.xscale('log')  # Escala logarítmica para o eixo x (períodos)
    plt.xlabel('Período (dias)', fontsize=14)
    plt.ylabel('Coerência', fontsize=14)
    plt.title(title, fontsize=16)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()

# Exemplo: Calcular e plotar a coerência para um par de latitudes
latitudes = data.columns  # Lista de latitudes
lat1, lat2 = latitudes[2], latitudes[-1]  # Escolha duas latitudes para comparar

# Extraia as séries temporais para as latitudes escolhidas
signal1 = data[lat1].values
signal2 = data[lat2].values

# Calcule e plote a coerência em função do período
plot_coherence_period(signal1, signal2, fs=1.0, title=f'Coerência entre {lat1} e {lat2}')



####
# mapa de coerencias  ---->> UTILIZEI ESSE
###

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import coherence

# Função para calcular a coerência e converter frequência para período
def calculate_coherence_period(signal1, signal2, fs=1.0):
    f, Cxy = coherence(signal1, signal2, fs=fs, window='hann', nperseg=256)
    period = 1 / f  # Converter frequência para período (dias)
    
    # Remover valores de frequência zero (para evitar NaN em período)
    valid_frequencies = f > 0
    period = period[valid_frequencies]
    Cxy = Cxy[valid_frequencies]
    
    return period, Cxy

# Lista de latitudes
latitudes = data.columns

for central_lat in latitudes:
# Escolha do ponto central
# central_lat = -33.000000  # Substitua pelo ponto central desejado
    central_index = np.where(latitudes == central_lat)[0][0]

    # Extrair o sinal do ponto central
    central_signal = data[central_lat].values

    # Inicializar a matriz de coerência
    n_latitudes = len(latitudes)
    n_periods = 128  # Número de períodos (ajuste conforme necessário)
    coherence_matrix = np.zeros((n_latitudes, n_periods))
    periods = np.zeros(n_periods)

    # Calcular a coerência entre o ponto central e todos os outros pontos
    for i, lat in enumerate(latitudes):
        if lat != central_lat:  # Ignorar o ponto central
            signal = data[lat].values
            period, Cxy = calculate_coherence_period(central_signal, signal, fs=1.0)
            
            # Preencher a matriz de coerência com os valores válidos
            coherence_matrix[i, :len(Cxy)] = Cxy[:n_periods]  # Garantir o mesmo tamanho
            periods[:len(period)] = period[:n_periods]  # Períodos correspondentes

    # Verificar valores não finitos
    if np.any(np.isnan(coherence_matrix)) or np.any(np.isnan(periods)):
        print("Existem valores NaN nos dados de coerência ou períodos!")

    # Plotar o gráfico
    plt.figure(figsize=(12, 8))
    cp = plt.contourf(periods, latitudes, coherence_matrix, np.arange(0,1.05,0.05), cmap='viridis')
    # Adicionar a barra de cores (colorbar)
    plt.colorbar(cp, label='Coerência')
    # plt.xscale('log')  # Escala logarítmica para o eixo x (períodos)
    plt.xlim([3, 30])
    plt.ylim([-33, 3.1])
    plt.xlabel('Período (dias)', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.title(f'Coerência em Relação a ({central_lat})', fontsize=16)
    plt.grid(True, which="both", ls="--", color='white', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'/Users/breno/mestrado/coerencia/{central_lat}.png')

'''
TODO:
1 - Fazer o calculo da velocidade utilizando a media
'''





#
# crosspecs <-
#
latitudes = data.columns
for i1 in data.columns:
    xx1=np.asarray(data[i1])
    for i2 in data.columns:
        xx2=np.asarray(data[i2])
        ppp=len(xx1)
        dt=24#diario
        win=2
        smo=999
        ci=99
        h1,h2,fff,coef,conf,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

        fig = plt.figure(figsize=(8,6))
        plt.plot(1./fff/24,coef,'b')
        plt.plot(1./fff/24,conf,'--k')
        plt.xlim([0,30])
        plt.ylabel('[Coerência]')
        plt.yticks([0,.5,1])
        plt.xlabel('Período (dias)')
        plt.grid()
        # plt.show()

        os.makedirs(f'/Users/breno/mestrado/crosspecs_coh/{i1}/', exist_ok=True)
        plt.savefig(f'/Users/breno/mestrado/crosspecs_coh/{i1}/{i2}.png')



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
