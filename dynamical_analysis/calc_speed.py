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

## alterar aqui pra fazer um hovmoller medio

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
##############
##############
##############
#######
#######
#######
#######
#######
#######
def compute_psd(ts, fs=1/1):  # fs = 1/day
    f, psd = signal.welch(ts, fs=fs, nperseg=365)  # 1-year segments
    periods = 1 / f  # Convert frequency to period (days)
    return periods, psd

# Compute PSD for all latitudes
periods_list, psd_data = [], {}
latitudes = data.index

for lat in latitudes:
    periods, psd = compute_psd(np.asarray(data.loc[lat]))
    psd_data[lat] = psd
# periods = periods[(periods >= 3) & (periods <= 30)]  # Filter to 5-30 days

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
cmap = plt.cm.viridis
for i, lat in enumerate(latitudes):
    psd = psd_data[lat]# [(periods >= 3) & (periods <= 30)]
    ax.plot(periods, psd, label=lat, color=cmap(i/len(latitudes)))

ax.set_xlabel('Period (days)')
ax.set_ylabel('Spectral Power [cm²/cph]')
ax.set_xlim(3, 30)
ax.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



########
latitudes = data.index

def compute_psd_matrix(ssh_data, fs=1/1):
    periods_list, psd_matrix = [], []
    for lat in latitudes:
        f, psd = signal.welch(ssh_data.loc[lat], fs=fs, nperseg=365)
        periods = 1 / f  # Convert frequency to period (days)
        psd_matrix.append(psd)
        periods_list = periods  # All periods are the same
    # Filter periods to 5-30 days
    period_mask = (periods_list >= 3) & (periods_list <= 30)
    psd_matrix = np.array(psd_matrix)[:, period_mask]
    periods = periods_list[period_mask]
    return periods, psd_matrix

periods, psd_matrix = compute_psd_matrix(data)


fig, ax = plt.subplots(figsize=(10, 6))

# Convert latitudes to numerical values (e.g., [-10, -15, ...])
lat_values = latitudes

# Create a grid for pcolormesh
X, Y = np.meshgrid(periods, lat_values)

# Plot heatmap
pc = ax.contour(X, Y, psd_matrix, shading='auto', cmap='viridis')
fig.colorbar(pc, ax=ax, label='Spectral Power [cm²/cph]')

ax.set_xlabel('Period (days)')
ax.set_ylabel('Latitude')
ax.set_title('Spectral Power')
plt.show()


####
# Example pairs (replace with your actual pairs)
pairs = [
    (-33.0, -15.008333, 'SSA vs AB'),
]

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# Coherence
ax = axs[0]
for i, (lat1, lat2, label) in enumerate(pairs):
    f, coh = signal.coherence(data.loc[lat1], data.loc[lat2], fs=1/1, nperseg=365)
    periods_coh = 1 / f
    period_mask = (periods_coh >= 5) & (periods_coh <= 30)
    ax.plot(periods_coh[period_mask], coh[period_mask], label=label)
ax.set_ylabel('Coherence Coefficient')
ax.legend()

# Lag
ax = axs[1]
for i, (lat1, lat2, label) in enumerate(pairs):
    f, coh, phase = signal.coherence(data.loc[lat1], data.loc[lat2], fs=1/1, nperseg=365, return_phase=True)
    lag = (phase / (2 * np.pi)) * (1 / f)  # Convert phase to lag (days)
    periods_lag = 1 / f
    period_mask = (periods_lag >= 5) & (periods_lag <= 30)
    ax.plot(periods_lag[period_mask], lag[period_mask], label=label)
ax.set_ylabel('Lag (days)')
ax.legend()

# Finalize
axs[-1].set_xlabel('Period (days)')
for ax in axs:
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(5, 30)
plt.tight_layout()

###################

# Função modificada para Spectral Power
def compute_psd_matrix(ssh_data, fs=1/1):
    periods_list, psd_matrix = [], []
    for lat in latitudes:
        f, psd = signal.welch(ssh_data.loc[lat], fs=fs, nperseg=365)
        periods = 1 / f  # Converter frequência para período (dias)
        psd_matrix.append(psd)
        periods_list = periods
    # Filtrar períodos e converter para matriz 2D
    period_mask = (periods_list >= 1) & (periods_list <= 40)
    psd_matrix = np.array(psd_matrix)[:, period_mask]
    periods = periods_list[period_mask]
    return periods, psd_matrix

# Plot com escala logarítmica
periods, psd_matrix = compute_psd_matrix(data)
lat_values = latitudes  # Converter para valores numéricos

X, Y = np.meshgrid(periods, lat_values)

fig, ax = plt.subplots(figsize=(10, 6))
levels =np.array([.1, 1, 20,  40,  60,  80,  100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
         1e3, 1.5e3, 2e3])
# Usar escala logarítmica no eixo X

# BDmatrixq = interp2(X,Y,psd_matrix,xq,yq,'cubic')


pc = ax.contourf(X, Y, psd_matrix,interpolation = 'linear', cmap='inferno_r', levels=levels,
                 norm=matplotlib.colors.LogNorm(vmin=.1, vmax=2e3))  # Ajustar vmin/vmax conforme seus dados)  # Ajustar vmin/vmax conforme seus dados

# ax.set_xscale('log')  # Escala log no eixo X
ax.set_xticks([3, 12, 16, 30])
ax.set_xlim([3,30])
# ax.get_xaxis().set_major_formatter(ScalarFormatter())  # Forçar labels não-científicas
plt.colorbar(pc, ax=ax, label='Spectral Power [cm²/cph]')
ax.set_xlabel('Period (days)')
ax.set_ylabel('Latitude')
plt.show()

# OK, AGORA FICOU MUITO DECENTE!

######crosspecs
latitudes = data.index


def coherence_crosspecs(central_lat):

    coherence_matrix = np.zeros((37, 290))
    lag_matrix = np.zeros_like(coherence_matrix)

    xx1=np.asarray(data.loc[central_lat])
    for i, lat in enumerate(latitudes):
        xx2=np.asarray(data.loc[lat])
        ppp=len(xx1)
        dt=24 #diario
        win=2
        smo=999
        ci=99
        h1,h2,fff,coef,conf,fase=crosspecs.crospecs(xx1, xx2, ppp, dt, win, smo, ci)

        coef = np.where(coef>conf, coef, np.nan)
        fase = np.where(coef>conf, fase, np.nan)

        periods = 1./fff/24


        coherence_matrix[i,:] = coef
        lag_matrix[i,:] = (fase/(2*np.pi) * (1/fff/24))

    return periods, coherence_matrix, lag_matrix


p2, c2, l2 = coherence_crosspecs( -25.00303)

central_lat = -25

# Criar grids para pcolormesh
X, Y = np.meshgrid(p2, latitudes)

lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

# Plotar Coerência
fig, ax = plt.subplots(figsize=(10, 4))
pc = ax.contourf(X, Y, c2, shading='auto', cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)
fig.colorbar(pc, ax=ax, label='Coherence Coefficient')
# ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 30])
ax.set_xlim([3,30])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Period (days)')
ax.axhline(central_lat, color='black', linestyle='--', linewidth=2)
ax.set_ylabel(f'Latitude (vs {central_lat})')
ax.set_title('Coerência')
plt.show()


X, Y = np.meshgrid(p2, latitudes)

lvllag = np.arange(-10, 15.5, .5)

lvlcontour = [-2, -.5, .5, 2, 4, 6, 8, 10, 12, 14, 16]

# Plotar Coerência
fig, ax = plt.subplots(figsize=(10, 4))
pc = ax.contourf(X, Y, l2, shading='auto', cmap='viridis', levels = lvllag)
# contours = ax.contour(X, Y, l2, levels=lvlcontour, colors='black', linewidths=.5)

# ax.clabel(contours, inline=True,fontsize=10)

fig.colorbar(pc, ax=ax, label='Lag Coefficient')
# ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 30])
ax.set_xlim([3,30])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Period (days)')
ax.axhline(central_lat, color='black', linestyle='--', linewidth=2)
ax.set_ylabel(f'Latitude (vs {central_lat})')
ax.set_title('Lag')
plt.show()









# combinando os 3

for central_lat in latitudes:


    p2, c2, l2 = coherence_crosspecs( central_lat)

    periods, psd_matrix = compute_psd_matrix(data)



    # Criar grids para pcolormesh

    fig, ax = plt.subplots(1,3, figsize=(18,15))

    ################ -> espectroX


    lvl_spec = np.array([.1, 1, 20,  40,  60,  80,  100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
            1e3, 1.5e3, 2e3])

    X, Y = np.meshgrid(periods, latitudes)


    spc = ax[0].contourf(X, Y, psd_matrix,interpolation = 'linear', cmap='inferno_r', levels=lvl_spec,
                    norm=matplotlib.colors.LogNorm(vmin=.1, vmax=2e3))  # Ajustar vmin/vmax conforme seus dados)  # Ajustar vmin/vmax conforme seus dados


    # Custom formatter para 10¹, 10²
    def exp_formatter(x, pos):
        exponent = int(np.log10(x))
        return r'$10^{%d}$' % exponent

    cbar_spec = plt.colorbar(spc, ax=ax[0], location='top', pad=0.02)
    cbar_spec.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(exp_formatter))
    cbar_spec.ax.xaxis.set_ticks([1e-1, 1e1, 1e2, 1e3])  # Valores exatos para corresponder aos exponents
    cbar_spec.ax.xaxis.set_ticks_position('bottom')


    # ax.set_xscale('log')  # Escala log no eixo X
    ax[0].set_xticks([3, 12, 16, 30])
    ax[0].set_xlim([3,30])
    # ax.get_xaxis().set_major_formatter(ScalarFormatter())  # Forçar labels não-científicas
    # plt.colorbar(spc, ax=ax, label='Spectral Power [cm²/cph]')
    ax[0].set_xlabel('Period (days)')
    ax[0].set_title('Spectral Power (cm²/cph)', pad=42)
    # ax[1].set_ylabel('Latitude')


    ################## ->coerencia

    X, Y = np.meshgrid(p2, latitudes)


    lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

    coc = ax[1].contourf(X, Y, c2, shading='auto', cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)

    # Colorbar no topo
    cbar_coh = plt.colorbar(coc, ax=ax[1], location='top', pad=0.02)
    # cbar_coh.set_label('Coherence', labelpad=10)
    # cbar_coh.ax.xaxis.set_label_position('bottom')
    cbar_coh.ax.xaxis.set_ticks_position('bottom')
    cbar_coh.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])


    # plt.colorbar(coc, ax=ax, label='Coherence Coefficient')
    # ax.set_xscale('log')
    ax[1].set_xticks([5, 10, 20, 30])
    ax[1].set_xlim([3,30])
    ax[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1].set_xlabel('Period (days)')
    ax[1].axhline(central_lat, color='black', linestyle='--', linewidth=2)
    ax[1].set_title('Coherence Coefficient', pad=42)



    ################# - > LAG


    lvllag = np.arange(-12, 15.5, .5)

    lvlcontour = [-2, 0, 1, 2, 4, 6, 8, 10, 12, 14, 16]

    col = ax[2].contourf(X, Y, l2, shading='auto', cmap='viridis', levels = lvllag)
    contours = ax[2].contour(X, Y, l2, levels=lvlcontour, colors='black', linewidths=0.7)
    ax[2].clabel(contours, inline=True, fontsize=12, colors = 'black',  fmt='%1.0f')

    # Colorbar no topo
    cbar_lag = plt.colorbar(col, ax=ax[2], location='top', pad=0.02)
    # cbar_lag.set_label('Lag (days)', labelpad=10)
    # cbar_lag.ax.xaxis.set_label_position('bottom')
    cbar_lag.ax.xaxis.set_ticks_position('bottom')
    cbar_lag.set_ticks(np.arange(-10, 16, 5))
    # contours = ax.contour(X, Y, l2, levels=lvlcontour, colors='black', linewidths=.5)

    # ax.clabel(contours, inline=True,fontsize=10)

    # plt.colorbar(col, ax=ax, label='Lag Coefficient')
    # ax.set_xscale('log')
    ax[2].set_xticks([5, 10, 20, 30])
    ax[2].set_xlim([3,30])
    ax[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[2].set_xlabel('Period (days)')
    ax[2].axhline(central_lat, color='black', linestyle='--', linewidth=2)
    ax[2].set_title('Lag', pad=42)


    plt.tight_layout()
    plt.savefig('/Users/breno/mestrado/triple_figure/' + str(central_lat) + '.png', dpi=300)

########

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.colors import LogNorm, Normalize

def compute_coherence_lag_matrices(central_lat, other_lats, data, fs=1/1, nperseg=365):
    """
    Retorna matrizes 2D de coerência e lag para a latitude central vs outras latitudes.
    """
    # Definir períodos de referência (garantir consistência)
    f_ref, _ = signal.welch(data.loc[central_lat], fs=fs, nperseg=nperseg)
    periods = 1 / f_ref
    period_mask = (periods >= 3) & (periods <= 40)
    periods = periods[period_mask]
    
    # Inicializar matrizes
    coherence_matrix = np.zeros((len(other_lats), len(periods)))
    lag_matrix = np.zeros_like(coherence_matrix)

    # Calcular para cada latitude
    for i, lat in enumerate(other_lats):
        f, Pxy = signal.csd(data.loc[central_lat], data.loc[lat], fs=fs, nperseg=nperseg)
        Pxx = signal.welch(data.loc[central_lat], fs=fs, nperseg=nperseg)[1]
        Pyy = signal.welch(data.loc[lat], fs=fs, nperseg=nperseg)[1]
        coherence = np.abs(Pxy)**2 / (Pxx * Pyy)
        phase = np.angle(Pxy)
        lag = (phase / (2 * np.pi)) * (1/f)
        
        # Filtrar e armazenar
        coherence_matrix[i, :] = coherence[period_mask]
        lag_matrix[i, :] = lag[period_mask]
    
    return periods, coherence_matrix, lag_matrix


# Definir latitude central e outras latitudes
central_lat = -33.0
other_lats =  [lat for lat in latitudes if lat != central_lat]

# Calcular matrizes
periods, coherence_matrix, lag_matrix = compute_coherence_lag_matrices(
    central_lat, latitudes, data
)

# Converter latitudes para valores numéricos (y-axis)
lat_values = latitudes

# Criar grids para pcolormesh
X, Y = np.meshgrid(periods, lat_values)

# Plotar Coerência
fig, ax = plt.subplots(figsize=(10, 4))
pc = ax.contourf(X, Y, coherence_matrix, shading='auto', cmap='viridis', vmin=0, vmax=1)
fig.colorbar(pc, ax=ax, label='Coherence Coefficient')
# ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 30])
ax.set_xlim([3,30])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Period (days)')
ax.set_ylabel(f'Latitude (vs {central_lat})')
ax.set_title('Coerência')
plt.show()

# Plotar Lag
fig, ax = plt.subplots(figsize=(10, 4))
pc = ax.pcolormesh(X, Y, lag_matrix, shading='auto', cmap='coolwarm', norm=Normalize(vmin=-5, vmax=5))
fig.colorbar(pc, ax=ax, label='Lag (days)')
ax.set_xscale('log')
ax.set_xticks([5, 10, 20, 30])
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_xlabel('Period (days)')
ax.set_ylabel(f'Latitude (vs {central_lat})')
ax.set_title('Lag de Fase')
plt.show()

##############
##############
##############
#######
#######
#######
#######
#######
#######

######
# tentando calcular coerencia
#####
from scipy.signal import coherence

coh_data = pd.read_csv('/Users/breno/mestrado/hov_data.csv', index_col=0)

def plot_coherence(signal1, signal2, fs=1.0, title='Coerência'):
    f, Cxy = coherence(signal1, signal2, fs=fs, window='hann', nperseg=256)
    plt.figure()
    plt.plot(f, Cxy)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Coerência')
    plt.title(title)
    plt.show()

data = coh_data.T.copy()

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
