import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import itertools
import cartopy
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
import time
from scipy.fft import fft, fftfreq
path.append(
    '/Users/breno/Documents/Mestrado/tese/scripts'
)
import filtro
import le_dado
import le_reanalise

# faz a analise pro dado:

fig_folder = '/Users/breno/Documents/Mestrado/t1'
path_dado = '/Volumes/BRENO_HD/dados_mestrado/dados/'
path_reanalise = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/GLOR4/'

filename = 'Ilha Fiscal 2014.txt'
formato = 'csv'
nome = 'Ilha Fiscal'

ifi = le_dado.le_dado_csv(filename, path_dado)
le_dado.pega_fft(ifi.sea_level, 'Ilha Fiscal', fig_folder)

ifi_band_h = le_dado.roda_analise(filename, formato, nome, 'band',fig_folder=fig_folder,
                 dataini = None, datafim = None)

ifi_band = le_dado.reamostra_dado(ifi_band_h.data, ifi_band_h.time)

# a parte de alta e baixa acaba sendo a mesma coisa q tem q usar

fig_folder = '/Users/breno/Documents/Mestrado/tese/figs/apresentacao/'

ifi_baixa = le_dado.roda_analise(filename, formato, nome, 'low',fig_folder=fig_folder,
                 dataini = None, datafim = None)

fig_folder = '/Users/breno/Documents/Mestrado/tese/figs/apresentacao/'

ifi_to_filt_alta = le_dado.reamostra_dado(ifi_baixa.data, ifi_baixa.time)
ifi_alta = filtro.filtra_dados(ifi_to_filt_alta.data, ifi_to_filt_alta.time, nome,
                               fig_folder, 'high')

## usar o ifi_alta pra comparar

## comecando agora a mexer com dado de modelo

figs_folder = '/Users/breno/Documents/Mestrado/tese/figs/apresentacao/'

year = 2014
path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/HYCOM/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'HYCOM'
point_name = 'Ilha Fiscal'

reanalisys = le_reanalise.read_reanalisys(str(year), path_reanalise)

dimensions = le_reanalise.get_reanalisys_dims(reanalisys)

ti = reanalisys.time.values[0]
tf = reanalisys.time.values[-1]
hycom = le_reanalise.cut_reanalisys(reanalisys, ti, tf,
                                               if_lat, if_lon, dimensions)
# hycom = le_reanalise.roda_tudo(year, if_lat, if_lon, model_name, point_name,
#               path_reanalise = path_reanalise, figs_folder = figs_folder)
ssh_hycom = hycom.values
time_hycom = hycom.time.values


hycom_filt = filtro.filtra_dados(ssh_hycom, time_hycom, 'HYCOM', figs_folder,
                                 'band', True)

hycom_filt_low = filtro.filtra_dados(ssh_hycom, time_hycom, 'HYCOM_low', figs_folder,
                                 'low', True)

hycom_filt2 = filtro.filtra_dados(hycom_filt_low, time_hycom, 'HYCOM_high', figs_folder,
                                 'high', True)

plt.plot(time_hycom, ssh_hycom, label = 'puro')
plt.plot(time_hycom, hycom_filt, label = 'banda')
plt.plot(time_hycom, hycom_filt2, label = 'composto')
plt.legend()
plt.grid()

#### pegando agora as correlacoes:
ifih_time = ifi_to_filt_alta.time

cbb = np.corrcoef(hycom_filt, ifi_band)
ccb = np.corrcoef(hycom_filt, ifi_alta)
cbc = np.corrcoef(hycom_filt2, ifi_band)
ccc = np.corrcoef(hycom_filt2, ifi_alta)

fig= plt.figure(figsize=(16,4))
trans = fig.transFigure
plt.title('Dado Medido (BANDA) x Modelado (BANDA)')
plt.plot(ifi_band.time, ifi_band, label='ILHA FISCAL DIARIO')
plt.plot(time_hycom, hycom_filt*100, label='HYCOM')
plt.legend()
plt.text(0.60, 0.85, f'CORR: {str((cbb[0][1]*100).round(1))}%',
         transform = trans, color = 'red',
         weight = 'bold')
plt.grid()
plt.tight_layout()
plt.savefig(figs_folder + 'compara_banda_banda.png')

fig= plt.figure(figsize=(16,4))
trans = fig.transFigure
plt.title('Dado Medido (COMPOSTO) x Modelado (BANDA)')
plt.plot(ifih_time, ifi_alta, label='ILHA FISCAL DIARIO')
plt.plot(time_hycom, hycom_filt*100, label='HYCOM')
plt.legend()
plt.text(0.60, 0.85, f'CORR: {str((ccb[0][1]*100).round(1))}%',
         transform = trans, color = 'red',
         weight = 'bold')
plt.grid()
plt.tight_layout()
plt.savefig(figs_folder + 'compara_composto_banda.png')

fig= plt.figure(figsize=(16,4))
trans = fig.transFigure
plt.title('Dado Medido (BANDA) x Modelado (COMPOSTO)')
plt.plot(ifi_band.time, ifi_band, label='ILHA FISCAL DIARIO')
plt.plot(time_hycom, hycom_filt2*100, label='HYCOM')
plt.legend()
plt.text(0.60, 0.85, f'CORR: {str((cbc[0][1]*100).round(1))}%',
         transform = trans, color = 'red',
         weight = 'bold')
plt.grid()
plt.tight_layout()
plt.savefig(figs_folder + 'compara_banda_composto.png')

fig= plt.figure(figsize=(16,4))
trans = fig.transFigure
plt.title('Dado Medido (COMPOSTO) x Modelado (COMPOSTO)')
plt.plot(ifih_time, ifi_alta, label='ILHA FISCAL DIARIO')
plt.plot(time_hycom, hycom_filt2*100, label='HYCOM')
plt.legend()
plt.text(0.60, 0.85, f'CORR: {str((ccc[0][1]*100).round(1))}%',
         transform = trans, color = 'red',
         weight = 'bold')
plt.grid()
plt.tight_layout()
plt.savefig(figs_folder + 'compara_composto_composto.png')

plt.close('all')


'''
TODO: 
1 - tabela com:
        correlação
        variabilidade
        média
        desvio padrão
Modelo da tabela encontrado na pasta tab


VARS: 
ifi_alta = serie de dados de ilha fiscal, já reamostrado pra diário e filtrado na passa_alta
ifih_time = serie de datas da ifi_alta
hycom_filt2 = serie da saida do hycom ja filtrado com a passa-banda
time_hycom = serie de datas pro hycom
'''


year = 2014

# importa glor12

path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR12/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'GLOR12'
point_name = 'Ilha Fiscal'

reanalisys = le_reanalise.read_reanalisys(str(year), path_reanalise)

dimensions = le_reanalise.get_reanalisys_dims(reanalisys)

ti = reanalisys.time.values[0]
tf = reanalisys.time.values[-1]
glor12 = le_reanalise.cut_reanalisys(reanalisys, ti, tf,
                                               if_lat, if_lon, dimensions)
# hycom = le_reanalise.roda_tudo(year, if_lat, if_lon, model_name, point_name,
#               path_reanalise = path_reanalise, figs_folder = figs_folder)
ssh_glor12 = glor12.values
time_glor12 = glor12.time.values
tg12 = time_glor12 - np.timedelta64(12,'h')


glor12_filt = filtro.filtra_dados(ssh_glor12, time_glor12, 'glor12', figs_folder,
                                 'band', True)

glor12_filt_low = filtro.filtra_dados(ssh_glor12, time_glor12, 'glor12_low', figs_folder,
                                 'low', True)

glor12_filt2 = filtro.filtra_dados(glor12_filt_low, time_glor12, 'glor12_high', figs_folder,
                                 'high', True)

# importa glor4

path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR4/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'GLOR4'
point_name = 'Ilha Fiscal'

reanalisys = le_reanalise.read_reanalisys(str(year), path_reanalise)

dimensions = le_reanalise.get_reanalisys_dims(reanalisys)

ti = reanalisys.time.values[0]
tf = reanalisys.time.values[-1]
glor4 = le_reanalise.cut_reanalisys(reanalisys, ti, tf,
                                               if_lat, if_lon, dimensions)
# hycom = le_reanalise.roda_tudo(year, if_lat, if_lon, model_name, point_name,
#               path_reanalise = path_reanalise, figs_folder = figs_folder)
ssh_glor4 = glor4.values
time_glor4 = glor4.time.values


glor4_filt = filtro.filtra_dados(ssh_glor4, time_glor4, 'glor4', figs_folder,
                                 'band', True)

glor4_filt_low = filtro.filtra_dados(ssh_glor4, time_glor4, 'glor4_low', figs_folder,
                                 'low', True)

glor4_filt2 = filtro.filtra_dados(glor4_filt_low, time_glor4, 'glor4_high', figs_folder,
                                 'high', True)

# importa o BRAN - esse eu acho q vai ser um pouco mais difíciil devido ao jeito q eles classificam a latlon

path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/BRAN/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'BRAN'
point_name = 'Ilha Fiscal'

reanalisys = le_reanalise.read_reanalisys(str(year), path_reanalise)

# dimensions = le_reanalise.get_reanalisys_dims(reanalisys)
dimensions = {'time': 'Time', 'lat' : 'yt_ocean', 'lon' : 'xt_ocean'}

ti = reanalisys.Time.values[0]
tf = reanalisys.Time.values[-1]
bran = le_reanalise.cut_reanalisys(reanalisys, ti, tf,
                                               if_lat, if_lon, dimensions)
# hycom = le_reanalise.roda_tudo(year, if_lat, if_lon, model_name, point_name,
#               path_reanalise = path_reanalise, figs_folder = figs_folder)
ssh_bran = bran.values
time_bran = bran.Time.values

tb = time_bran - np.timedelta64(12,'h')

bran_filt = filtro.filtra_dados(ssh_bran, time_bran, 'bran', figs_folder,
                                 'band', True)

bran_filt_low = filtro.filtra_dados(ssh_bran, time_bran, 'bran_low', figs_folder,
                                 'low', True)

bran_filt2 = filtro.filtra_dados(bran_filt_low, time_bran, 'bran_high', figs_folder,
                                 'high', True)


### FAZ TABELA:
hy = hycom_filt2
g12 = glor12_filt2
g4 = glor4_filt2
br = bran_filt2

# calcula as medias -> EM CM:
if_mean = ifi_alta.mean()
h_mean = hy.mean()*100
g12_mean = g12.mean()*100

data = [ifi_alta, hy, g12, g4, br]
names = ['ILHA FISCAL', 'HYCOM', 'GLOR12', 'GLOR4', 'BRAN']
df_stats = pd.DataFrame()
df_stats.index = ['Média (cm)', 'Variância (cm²)', 'Corr. com dado(%)']

for i, name in zip(data, names):
    if i.max() <1:
        i = i * 100
    media= round(i.mean(),2)
    var = round(np.var(i), 2)
    var = round(np.var(i), 2)
    std = round(np.std(i), 2)
    corr =round(np.corrcoef(i, ifi_alta)[0,1] * 100, 2)

    df_stats[name] = [media, var,  corr]

df_stats = df_stats.T

df_stats.to_latex

data = [ifi_alta, hy, g12, g4, br]

for i, name in zip(data, names):
    print(name+' alta')
    print(round(np.corrcoef(i, ifi_band)[0,1] * 100, 2), round(np.corrcoef(i, ifi_alta)[0,1] * 100, 2))

print('------------')
data1 = [ifi_band, hycom_filt, glor12_filt, glor4_filt, bran_filt]

for i, name in zip(data1, names):
    print(name + ' band')
    print(round(np.corrcoef(i, ifi_band)[0,1] * 100, 2), round(np.corrcoef(i, ifi_alta)[0,1] * 100, 2))



##### NEW STATS:
data = [ifi_alta, hy, g12, g4, br]
spaos = [hy, g12, g4, br]
spaos_name = ['hy', 'g12', 'g4', 'br']

# corr = ifi_alta.corr(g12)  # modelo e dado
for i, n in zip(spaos, spaos_name):
    bias = np.mean(ifi_alta - i)
    rmse = np.sqrt(np.sum((i - ifi_alta)**2) /len(ifi_alta))
    si = rmse/np.mean(ifi_alta) 
    print(n)
    print('BIAS = ' + str(bias))
    print('RMSE = ' + str(rmse))
    print('SI = ' + str(si))
    
# bias = np.mean(results_modelo-dado_boia)  # modelo e dado
# rmse = np.sqrt(np.sum((results_modelo - dado_boia)**2) /len(dado_boia))  # modelo e dado
# si = rmse/np.mean(dado_boia)  # dado

##### plotar tudo

fig = plt.figure(figsize=(14,8))
fig.suptitle('Séries Temporais de Altura de Superfície do Mar para Ilha Fiscal e os SPAOs')
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax1= fig.add_subplot(411)
ax2= fig.add_subplot(412)
ax3= fig.add_subplot(413)
ax4= fig.add_subplot(414)


ax1.plot(time_hycom, hy*100, label = 'HYCOM')
ax1.plot(ifih_time, ifi_alta, label = 'Ilha Fiscal Diário')
ax1.grid()
ax1.legend(fontsize='x-small', loc='lower left')

ax2.plot(time_glor12, g12*100, label = 'GLOR12')
ax2.plot(ifih_time, ifi_alta, label = 'Ilha Fiscal Diário')
ax2.grid()
ax2.legend(fontsize='x-small', loc='lower left')

ax3.plot(time_glor4, g4*100, label = 'GLOR4')
ax3.plot(ifih_time, ifi_alta, label = 'Ilha Fiscal Diário')
ax3.grid()
ax3.legend(fontsize='x-small', loc='lower left')

ax4.plot(time_bran, br*100, label = 'BRAN')
ax4.plot(ifih_time, ifi_alta, label = 'Ilha Fiscal Diário')
ax4.grid()
ax4.legend(fontsize='x-small', loc='lower left')

ax.set_ylabel('(cm)')
# plt.show()
plt.tight_layout()

plt.savefig(fig_folder + 'compara', dpi=200)

#######################################################################
### plotar comparação dos espectros de energia
#######################################################################

fig = plt.figure(figsize=(14,8))
fig.suptitle('Espectros de Energia para Ilha Fiscal e os SPAOs')
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax1= fig.add_subplot(411)
ax2= fig.add_subplot(412)
ax3= fig.add_subplot(413)
ax4= fig.add_subplot(414)



axes = [ax1, ax2, ax3, ax4]
sshs = [hy*100, g12*100, g4*100, br*100]
names = ['HYCOM', 'GLOR12', 'GLOR4', 'BRAN']

N = 365 # numero de medicoes
T = 1.0 # frequencia das medicoes (nesse caso = 1 medicao a cada 24h)

yif = fft(ifi_alta)
xif = fftfreq(N,T)
tif = 1/xif


for (axe, ssh, name) in zip(axes, sshs, names):

    yf = fft(ssh) # faz a transformada de fourier
    xf = fftfreq(N, T) # freequencia de amostragem da transformada
    # por algum motivo, o exemplo usa esse xfa, mas eu nao achei que valeria a pena
    # xfa = xf[:N//2]
    # tfa = 1/xfa
    tf = 1/xf

    # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    # ax1.semilogx(tfa, 2.0/N * np.abs(yf[0:N//2]))
    # ax1.set_ylim([-0.1, 40])
    # ax1.grid()
    axe.semilogx(tf, 2.0/N * np.abs(yf), label = name)
    axe.semilogx(tif, 2.0/N * np.abs(yif), label = 'Ilha Fiscal')
    axe.set_xlim(2.5,45)
    # TODO: CONSERTAR ISSO AQUI EMBAIXO!
    axe.set_xticks([3, 5, 10, 20, 30, 40]) 

    axe.set_xticklabels([])
    axe.legend()
    # axe.set_yticks([])

    # axe.ylim([-0.1, 40])
    axe.grid()

axe.set_xticklabels([3, 5, 10, 20, 30, 40])

ax.set_ylabel('Densidade de Altura de Superfície do Mar por faixa de frequência (m)')
ax.set_xlabel('Dias')
plt.tight_layout()
plt.savefig(fig_folder + 'compara_espectro', dpi=200)



############### 
# FAZ PLOT DO MAPA DOS 4
##############

path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR4/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'GLOR4'
point_name = 'Ilha Fiscal'
g4r = le_reanalise.read_reanalisys(str(year), path_reanalise)
# dimensionsg4 = le_reanalise.get_reanalisys_dims(g4r)
latsg4, lonsg4 = g4r.latitude.values, g4r.longitude.values

path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR12/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'GLOR12'
point_name = 'Ilha Fiscal'
g12r = le_reanalise.read_reanalisys(str(year), path_reanalise)
# dimensionsg12 = le_reanalise.get_reanalisys_dims(g12r)
latsg12, lonsg12 = g12r.latitude.values, g12r.longitude.values


path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/HYCOM/'
# if_lat = -22.897
# if_lon = -43.16500000000002fil
if_lat = -23.04
if_lon = -43.12
model_name = 'HYCOM'
point_name = 'Ilha Fiscal'
hycomr = le_reanalise.read_reanalisys(str(year), path_reanalise)
# dimensionshy = le_reanalise.get_reanalisys_dims(hycomr)
latshy, lonshy = hycomr.lat.values, hycomr.lon.values


path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/BRAN/'
# if_lat = -22.897
# if_lon = -43.1650000000000
lon = -43.165
lat = -22.897

if_lat = -23.04
if_lon = -43.12
model_name = 'BRAN'
point_name = 'Ilha Fiscal'
branr = le_reanalise.read_reanalisys(str(year), path_reanalise)
# dimensionsbra = le_reanalise.get_reanalisys_dims(branr)
latsbra, lonsbra = branr.yt_ocean.values, branr.xt_ocean.values

############
#############
# QUE BRAGUNCA HEIN! PQP!!!!
##############
##########

fig = plt.figure(figsize=(10,8))
proj = ccrs.PlateCarree()
ax1= fig.add_subplot(221, projection=proj)
ax2= fig.add_subplot(222, projection=proj)
ax3= fig.add_subplot(223, projection=proj)
ax4= fig.add_subplot(224, projection=proj)

axes = [ax1, ax2, ax3, ax4]
names = ['HYCOM', 'GLOR12', 'GLOR4', 'BRAN']
lones = (lonshy, lonsg12, lonsg4, lonsbra)
lates = (latshy, latsg12, latsg4, latsbra)

latlon = [(-23.04,-43.12),(-23,-43.08),(-23,-43),(-23.05,-43.15)]

nome_ponto = 'Ilha Fiscal'

if_lat = -22.897
if_lon = -43.1650000000000

for axe, nome_modelo, lons, lats, latelone in zip(axes, names, lones, lates, latlon):
    # proj = ccrs.PlateCarree()

    # fig, axe = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lat = latelone[0]
    lon = latelone[1]
    lon_max, lon_min, lat_max, lat_min = lon + 0.5 , lon - 0.5, lat + 0.5, lat - 0.5
    axe.set_extent([lon_max, lon_min, lat_max, lat_min], crs=ccrs.PlateCarree())

    axe.add_feature(cfeature.LAND, facecolor='0.3')
    axe.add_feature(cfeature.LAKES, alpha=0.9)  
    axe.add_feature(cfeature.BORDERS, zorder=10)
    axe.add_feature(cfeature.COASTLINE, zorder=10)

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',  name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    axe.add_feature(states_provinces, edgecolor='black', zorder=10) 

    axe.plot(if_lon, if_lat,
            color='red', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            # ,label = 'Ponto da Ilha Fiscal'
            )  
    # axe.text(lon - 0.005, lat - 0.005, nome_ponto,
    #       horizontalalignment='right', color = 'red', weight = 'bold',
    #       transform=ccrs.PlateCarree())  

    axe.plot(lon, lat,
            color='green', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            # ,label = 'Ponto utilizado'
            )
# # to_remove
    # axe.text(-43.2 - 0.005, -22.88 - 0.005, 'ponto mais proximo hycom (s/ dados)',
    #       horizontalalignment='right', color = 'red', weight = 'bold',
    #       transform=ccrs.PlateCarree())
    # axe.plot(-43.12, -22.96,
    #         color='red', linewidth=2, marker='o',
    #         transform=ccrs.PlateCarree()
    #         )
    # axe.text(-43.12 - 0.005, -22.96 - 0.005, 'ponto sem dados',
    #       horizontalalignment='right', color = 'red', weight = 'bold',
    #       transform=ccrs.PlateCarree())
    # axe.plot(-43.12, -23.04,
    #         color='green', linewidth=2, marker='o',
    #         transform=ccrs.PlateCarree()
    #         )
    # axe.text(-43.12 - 0.005, -23.04 - 0.005, 'ponto mais proximo com dados',
    #       horizontalalignment='right', color = 'green', weight = 'bold',
    #       transform=ccrs.PlateCarree())


    gl = axe.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                      color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    axe.set_title(f'{nome_modelo}')


fig.legend(['Ponto Ilha Fiscal', 'Ponto Utilizado'], loc='center')
plt.tight_layout()
plt.savefig(fig_folder + 'compara_mapas', dpi=200)
plt.close('all')