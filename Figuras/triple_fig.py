import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from scipy import signal
import xarray as xr

from sys import path
path.append('../old')


import crosspecs

model_path = '/Users/breno/mestrado/REANALISES_TEMP/BRAN/'

data = pd.read_csv('/Users/breno/mestrado/hov_data.csv', index_col=0)

latitudes = data.index


# def get_reanalisys(lat, lon, model, di, df):
#     reanal = {}
#     years = list(set([di.year, df.year]))
#     for year in years:
#         reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + str(year)  + '/*.nc')
#                                            , model)        
#         # reanal[year] = set_reanalisys_dims(xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
#         #                                      , model)
        
#     reanalisys = xr.concat(list(reanal.values()), dim="time")
#     model_series = reanalisys.sel(latitude=lat, longitude=lon, method='nearest')
#     model_series = model_series.sel(time=slice(di, df))

#     mod_ssh = model_series['ssh'].values
#     # nao tava dando problema entao n ha necessidade de fazer assim
#     # mod_ssh_to_filt = np.concatenate((np.full(len(mod_ssh)//2, mod_ssh.mean()), mod_ssh))
#     # mod_time = pd.date_range(end=model_series['time'].values[-1], periods=len(mod_ssh_to_filt), freq='D')
#     mod_time = model_series['time'].values

#     return reanalisys, mod_ssh,  mod_time


# def get_points():
#     pts = {'lon': {
#   0: -51.74,
#   1: -51.21,
#   2: -50.42,
#   3: -49.675,
#   4: -49.056746,
#   5: -48.460366,
#   6: -48.24693,
#   7: -47.875,
#   8: -46.825,
#   9: -45.375,
#   10: -42.541667,
#   11: -40.425,
#   12: -40.281667,
#   13: -39.862634,
#   14: -38.908333,
#   15: -37.358333,
#   16: -38.690000,
#   17: -37.975,
#   18: -38.843861,
#   19: -38.850855,
#   20: -38.441667,
#   21: -37.477888,
#   22: -36.918087,
#   23: -35.753296,
#   24: -34.8866,
#   25: -34.550997,
#   26: -34.555217,
#   27: -34.958333,
#   28: -35.099074,
#   29: -37.512975,
#   30: -38.877564,
#   31: -42.870000,
#   32: -43.604459,
#   33: -44.641667,
#   34: -46.941667,
#   35: -48.375,
#   36: -49.508333},
#  'lat': {0: -33.0,
#   1: -32.0,
#   2: -31.0,
#   3: -30.003125,
#   4: -29.008333,
#   5: -28.008333,
#   6: -27.008333,
#   7: -26.008333,
#   8: -25.00303,
#   9: -24.000177,
#   10: -23.000088,
#   11: -22.00119,
#   12: -21.008333,
#   13: -20.008333,
#   14: -19.004932,
#   15: -18.008333,
#   16: -17.00718,
#   17: -16.006764,
#   18: -15.008333,
#   19: -14.008333,
#   20: -13.002109,
#   21: -12.008333,
#   22: -11.008333,
#   23: -10.008333,
#   24: -9.0083333,
#   25: -8.0083333,
#   26: -7.0083333,
#   27: -6.0024336,
#   28: -5.0083333,
#   29: -4.0083333,
#   30: -3.0083333,
#   31: -2.0009244,
#   32: -1.0083333,
#   33: -0.0027777778,
#   34: 1.0029762,
#   35: 2.0008772,
#   36: 3.0053483}}
#     return pd.DataFrame(pts)



# def collect_ssh_data(pts, di, df, model):
#     ssh_data = []
#     lats = []
#     times = []
    
#     for index, row in pts.iterrows():
#         lat = row['lat']
#         lon = row['lon']
#         _, reanalisys, fil_reanalisys, filt_time = get_reanalisys(lat, lon, model, di, df)
        
#         # Armazenar latitude, tempo e dados de SSH
#         lats.extend([lat] * len(filt_time))
#         times.extend(filt_time)
#         ssh_data.extend(fil_reanalisys)
    
#     # Criar DataFrame para os dados
#     df_ssh = pd.DataFrame({
#         'time': times,
#         'lat': lats,
#         'ssh': ssh_data
#     })
    
#     return df_ssh


def compute_psd_matrix(ssh_data, fs=1/1):
    periods_list, psd_matrix = [], []
    for lat in latitudes:
        f, psd = signal.welch(ssh_data.loc[lat], fs=fs, nperseg=365)
        periods = 1 / f  # Convert frequency to period (days)
        psd_matrix.append(psd)
        periods_list = periods  # All periods are the same
    # Filter periods to 5-30 days
    period_mask = (periods_list >= 3) & (periods_list <= 40)
    psd_matrix = np.array(psd_matrix)[:, period_mask]
    periods = periods_list[period_mask]
    return periods, psd_matrix


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


def exp_formatter(x, pos):
    exponent = int(np.log10(x))
    return r'$10^{%d}$' % exponent

# combinando os 3

for central_lat in latitudes:


    p2, c2, l2 = coherence_crosspecs( central_lat)


    periods, psd_matrix = compute_psd_matrix(data)



    # Criar grids para pcolormesh

    fig, ax = plt.subplots(1,4, figsize=(20,15))

    ax[0].tick_params(axis='both', labelsize=18)
    ax[1].tick_params(axis='both', labelsize=18)
    ax[2].tick_params(axis='both', labelsize=18)


    ################ -> espectroX


    lvl_spec = np.array([.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
            1e3, 1.5e3, 2e3])

    X, Y = np.meshgrid(periods, latitudes)


    spc = ax[0].contourf(X, Y, psd_matrix,interpolation = 'linear', cmap='inferno_r', levels=lvl_spec,
                    norm=matplotlib.colors.LogNorm(vmin=.1, vmax=2e3))  # Ajustar vmin/vmax conforme seus dados)  # Ajustar vmin/vmax conforme seus dados

    cbar_spec = plt.colorbar(spc, ax=ax[0], location='top', pad=0.035)
    cbar_spec.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(exp_formatter))
    cbar_spec.ax.xaxis.set_ticks([1e1, 1e2, 1e3], fontsize=18)  # Valores exatos para corresponder aos exponents
    cbar_spec.ax.xaxis.set_ticks_position('bottom')

    cbar_spec.ax.tick_params(labelsize=18)


    # ax.set_xscale('log')  # Escala log no eixo X
    ax[0].set_xticks([3, 12, 16, 30])
    ax[0].set_xlim([3,30])
    # ax.get_xaxis().set_major_formatter(ScalarFormatter())  # Forçar labels não-científicas
    # plt.colorbar(spc, ax=ax, label='Spectral Power [cm²/cph]')
    ax[0].set_ylabel('Latitude', fontsize=20)
    ax[0].set_xlabel('Period (days)', fontsize=20)
    ax[0].set_title('Spectral Power (cm²/cph)', pad=57, fontsize=24)
    # ax[1].set_ylabel('Latitude')


    ################## ->coerencia

    X, Y = np.meshgrid(p2, latitudes)


    lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

    coc = ax[1].contourf(X, Y, c2, cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)

    # Colorbar no topo
    cbar_coh = plt.colorbar(coc, ax=ax[1], location='top', pad=0.035)
    # cbar_coh.set_label('Coherence', labelpad=10)
    # cbar_coh.ax.xaxis.set_label_position('bottom')
    cbar_coh.ax.xaxis.set_ticks_position('bottom')
    cbar_coh.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    cbar_coh.ax.tick_params(labelsize=18)


    # plt.colorbar(coc, ax=ax, label='Coherence Coefficient')
    # ax.set_xscale('log')
    ax[1].set_xticks([5, 10, 20, 30])
    ax[1].set_xlim([3,30])
    ax[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1].set_xlabel('Period (days)', fontsize=20)
    ax[1].axhline(central_lat, color='black', linestyle='--', linewidth=2)
    ax[1].set_title('Coherence Coefficient', pad=57, fontsize = 24)



    ################# - > LAG


    lvllag = np.arange(-12, 15.5, .5)

    lvlcontour = [-2, 0, 1, 2, 4, 6, 8, 10, 12, 14, 16]

    col = ax[2].contourf(X, Y, l2, cmap='viridis', levels = lvllag)
    contours = ax[2].contour(X, Y, l2, levels=lvlcontour, colors='black', linewidths=0.7)
    ax[2].clabel(contours, inline=True, fontsize=18, colors = 'black',  fmt='%1.0f')

    # Colorbar no topo
    cbar_lag = plt.colorbar(col, ax=ax[2], location='top', pad=0.035)
    # cbar_lag.set_label('Lag (days)', labelpad=10)
    # cbar_lag.ax.xaxis.set_label_position('bottom')
    cbar_lag.ax.xaxis.set_ticks_position('bottom')
    cbar_lag.set_ticks(np.arange(-10, 16, 5))

    cbar_lag.ax.tick_params(labelsize=18)
    # contours = ax.contour(X, Y, l2, levels=lvlcontour, colors='black', linewidths=.5)

    # ax.clabel(contours, inline=True,fontsize=10)

    # plt.colorbar(col, ax=ax, label='Lag Coefficient')
    # ax.set_xscale('log')
    ax[2].set_xticks([5, 10, 20, 30])
    ax[2].set_xlim([3,30])
    ax[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[2].set_xlabel('Period (days)', fontsize=20)
    ax[2].axhline(central_lat, color='black', linestyle='--', linewidth=2)
    ax[2].set_title('Lag', pad=57, fontsize=24)


    # plt.tight_layout()
    # plt.savefig('/Users/breno/mestrado/key_fig.png', dpi=300)

    plt.savefig('/Users/breno/mestrado/triple_figure/' + str(central_lat) + '.png', dpi=300)



p2, c2, l2 = coherence_crosspecs(-33)

p3, c3, l3 = coherence_crosspecs(-22.00119)

p4, c4, l4 = coherence_crosspecs(-15.008333)

periods, psd_matrix = compute_psd_matrix(data)



# Criar grids para pcolormesh

fig, ax = plt.subplots(1,4, figsize=(20,15))

ax[0].tick_params(axis='both', labelsize=18)
ax[1].tick_params(axis='both', labelsize=18)
ax[2].tick_params(axis='both', labelsize=18)
ax[3].tick_params(axis='both', labelsize=18)



################ -> espectroX


lvl_spec = np.array([.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
        1e3, 1.5e3, 2e3])

X, Y = np.meshgrid(periods, latitudes)


spc = ax[0].contourf(X, Y, psd_matrix,interpolation = 'linear', cmap='inferno_r', levels=lvl_spec,
                norm=matplotlib.colors.LogNorm(vmin=.1, vmax=2e3))  # Ajustar vmin/vmax conforme seus dados)  # Ajustar vmin/vmax conforme seus dados

cbar_spec = plt.colorbar(spc, ax=ax[0], location='top', pad=0.035)
cbar_spec.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(exp_formatter))
cbar_spec.ax.xaxis.set_ticks([1e1, 1e2, 1e3], fontsize=18)  # Valores exatos para corresponder aos exponents
cbar_spec.ax.xaxis.set_ticks_position('bottom')

cbar_spec.ax.tick_params(labelsize=18)


# ax.set_xscale('log')  # Escala log no eixo X
ax[0].set_xticks([5, 10, 20, 30])
# ax[0].set_xticks([3, 12, 16, 30])
ax[0].set_xlim([3,30])
# ax.get_xaxis().set_major_formatter(ScalarFormatter())  # Forçar labels não-científicas
# plt.colorbar(spc, ax=ax, label='Spectral Power [cm²/cph]')
ax[0].set_ylabel('Latitude', fontsize=20)
ax[0].set_xlabel('Period (days)', fontsize=20)
ax[0].set_title('Spectral Power (cm²/cpd)', pad=57, fontsize=24)
# ax[1].set_ylabel('Latitude')


################## ->coerencia

X, Y = np.meshgrid(p2, latitudes)


lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

coc = ax[1].contourf(X, Y, c2, cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)

# Colorbar no topo
cbar_coh = plt.colorbar(coc, ax=ax[1], location='top', pad=0.035)
# cbar_coh.set_label('Coherence', labelpad=10)
# cbar_coh.ax.xaxis.set_label_position('bottom')
cbar_coh.ax.xaxis.set_ticks_position('bottom')
cbar_coh.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

cbar_coh.ax.tick_params(labelsize=18)


# plt.colorbar(coc, ax=ax, label='Coherence Coefficient')
# ax.set_xscale('log')
ax[1].set_xticks([5, 10, 20, 30])
ax[1].set_xlim([3,30])
ax[1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax[1].set_xlabel('Period (days)', fontsize=20)
ax[1].axhline(-33, color='black', linestyle='--', linewidth=2)
ax[1].set_title('Coherence Coefficient', pad=57, fontsize = 24)



################## ->coerencia

X, Y = np.meshgrid(p3, latitudes)


lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

coc = ax[2].contourf(X, Y, c3, cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)

# Colorbar no topo
cbar_coh = plt.colorbar(coc, ax=ax[2], location='top', pad=0.035)
# cbar_coh.set_label('Coherence', labelpad=10)
# cbar_coh.ax.xaxis.set_label_position('bottom')
cbar_coh.ax.xaxis.set_ticks_position('bottom')
cbar_coh.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

cbar_coh.ax.tick_params(labelsize=18)


# plt.colorbar(coc, ax=ax, label='Coherence Coefficient')
# ax.set_xscale('log')
ax[2].set_xticks([5, 10, 20, 30])
ax[2].set_xlim([3,30])
ax[2].get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax[2].set_xlabel('Period (days)', fontsize=20)
ax[2].axhline(-22.00119, color='black', linestyle='--', linewidth=2)
ax[2].set_title('Coherence Coefficient', pad=57, fontsize = 24)


################## ->coerencia

X, Y = np.meshgrid(p4, latitudes)


lvlcoh = [0, .1, .2, .3, .4, .5,  .6, .7,  .8, .9,  1]

coc = ax[3].contourf(X, Y, c4, cmap='viridis', vmin=0, vmax=1, levels=lvlcoh)

# Colorbar no topo
cbar_coh = plt.colorbar(coc, ax=ax[3], location='top', pad=0.035)
# cbar_coh.set_label('Coherence', labelpad=10)
# cbar_coh.ax.xaxis.set_label_position('bottom')
cbar_coh.ax.xaxis.set_ticks_position('bottom')
cbar_coh.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

cbar_coh.ax.tick_params(labelsize=18)


# plt.colorbar(coc, ax=ax, label='Coherence Coefficient')
# ax.set_xscale('log')
ax[3].set_xticks([5, 10, 20, 30])
ax[3].set_xlim([3,30])
ax[3].get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax[3].set_xlabel('Period (days)', fontsize=20)
ax[3].axhline(-15.008333, color='black', linestyle='--', linewidth=2)
ax[3].set_title('Coherence Coefficient', pad=57, fontsize = 24)



# plt.tight_layout()
plt.savefig('/Users/breno/mestrado/key_fig.png', dpi=300)

# plt.savefig('/Users/breno/mestrado/triple_figure/' + str(central_lat) + '.png', dpi=300)

