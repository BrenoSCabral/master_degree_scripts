home_path =  '/home/bcabral/' # set for Server

from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import numpy as np
import pandas as pd

from read_reanalisys import set_reanalisys_dims
from read_data import read_exported_series, treat_exported_series
from sys import path
path.append('old')
import filtro

def plot_grid_point(lat, lon, nome_modelo, nome_ponto, figs_folder, lons, lats, lat_mod, lon_mod):
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lon_max, lon_min, lat_max, lat_min = lon + 1 , lon - 1, lat + 1, lat - 1
    ax.set_extent([lon_max, lon_min, lat_max, lat_min], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='0.3')
    ax.add_feature(cfeature.LAKES, alpha=0.9)  
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',  name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', zorder=10) 


    plt.plot(lon, lat,
            color='red', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            )  
    plt.text(lon - 0.05, lat - 0.05, nome_ponto,
          horizontalalignment='right', color = 'red', weight = 'bold',
          transform=ccrs.PlateCarree())  

    
    plt.plot(lon_mod, lat_mod,
            color='green', linewidth=2, marker='o',
            transform=ccrs.PlateCarree()
            )    

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
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
    ax.set_title(f'Ponto {nome_ponto}')

    plt.tight_layout()
    plt.savefig(figs_folder + f'{nome_modelo}_{nome_ponto}_dominio.png')


######
def mapa_corr():
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon+0.4, lon-0.4, lat+0.4, lat-0.4], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.plot(lon, lat,
                color='red', linewidth=2, marker='o',
                transform=ccrs.PlateCarree()
                )  
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    longitude, latitude = np.meshgrid(reanal_subset.longitude, reanal_subset.latitude)



    c = ax.pcolormesh(longitude, latitude, correlation, transform=ccrs.PlateCarree(), cmap='coolwarm')

    cb = plt.colorbar(c, orientation='horizontal', pad=0.05)
    cb.set_label('Correlation')


    # c = ax.contourf(longitude, latitude, correlation, transform=ccrs.PlateCarree(), cmap='coolwarm', levels=20)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5,
                        color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    lons = reanal_subset['longitude'].values
    lats = reanal_subset['latitude'].values
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    # ax.set_title(f'Ponto {nome_ponto}')

    plt.tight_layout()

    plt.title('Correlation Map')

    plt.show()


'''
ideia:
- Acessar um modelo
- Acessar um ponto de dado
- comparar N pontos do modelo com este ponto de dado  (idealmente, usar uma grade ao redor do ponto)
- fazer mapa(s) de BIAS (e outras metricas estatisticas)
- Repetir para todos os modelos
'''


model_path = '/data3/MOVAR/modelos/REANALISES'
model_path = '/Users/breno/model_data/'
# modelos = ['BRAN', 'CGLO', 'ECCO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  ,'SODA']
modelos = ['BRAN']
model = 'BRAN'
data_path  = '/Users/breno/Documents/Mestrado/resultados/2014/data/'
results = {}
year = 2014


# for model in modelos:
fig_folder = '/Users/breno/Documents/Mestrado/resultados/2014/fig'
ifi = read_exported_series(data_path+'ilha_fiscal_2014.csv')
lat = ifi['lat'][0]
lon = ifi['lon'][0]
results[model] = {}
# path_model = f'/data3/MOVAR/modelos/REANALISES/{model}/SSH/{year}/'
path_model = '/Users/breno/model_data/BRAN/'

path_model = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/BRAN/2014/'
reanalisys = xr.open_mfdataset(path_model + '*.nc')
reanalisys = set_reanalisys_dims(reanalisys, model)

# lat_resolution = abs(float(reanalisys.latitude[0] - reanalisys.latitude[1]))
# lon_resolution = abs(float(reanalisys.longitude[0] - reanalisys.longitude[1]))

lat_idx = np.abs(reanalisys['latitude'] - lat).argmin().item()
lon_idx = np.abs(reanalisys['longitude'] - lon).argmin().item()


# Defina a grade ao redor do ponto (ex: 3x3 ao redor do ponto central)
# pegando 5x5
lat_indices = range(lat_idx-3, lat_idx+2)
lon_indices = range(lon_idx-2, lon_idx+3)
# lat_indices = range(lat_idx-8, lat_idx-1)
# lon_indices = range(lon_idx-3, lon_idx+4)
reanal_subset = reanalisys.isel(latitude=lat_indices, longitude=lon_indices)
# reference =  ifi[ifi.index.time == pd.to_datetime('03:00:00').time()]

# Vou utilizar o horário das 12h no dado, filtrar ele composto e pegar o resultado do modelo na banda pra fazer
# o estudo do ponto.

reference = ifi[::12] # dados horarios
# passa baixa -> reamostra -> passa alta
ifi_low = filtro.filtra_dados(reference['ssh'], reference.index, 'ifi', fig_folder, 'low')
ifi_low = ifi_low[::24]
ifi_high = filtro.filtra_dados(ifi_low, reference.index[::24], 'ifi', fig_folder, 'high')

ifi_filt = ifi_high


reanal_subset['ssh'].load()


# locsubset = reanal_subset.isel(latitude=0, longitude=0)
# mod_ssh = locsubset['ssh'].values
# mod_time = locsubset['time'].values


# 2 - passa banda -> Passa banda n ta funcionando muito legal nao

correlation = np.empty((len(reanal_subset.latitude), len(reanal_subset.longitude)))
for i in range(len(reanal_subset.latitude)):
    for j in range(len(reanal_subset.longitude)):
        model_series = reanal_subset.isel(latitude=i, longitude=j)
        mod_ssh = model_series['ssh'].values
        mod_time = model_series['time'].values
        mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'mod', fig_folder, 'band', modelo = True)
        correlation[i, j] = np.corrcoef(ifi_filt, mod_band)[0, 1]
correlation


#################
#################
# TESTES
reference = ifi[::12] # dados horarios
# passa baixa -> reamostra -> passa alta
ifi_low = filtro.filtra_dados(reference['ssh'], reference.index, 'ifi', fig_folder, 'low')
ifi_low0 = ifi_low[::24]
ifi_high0 = filtro.filtra_dados(ifi_low0, reference.index[::24], 'ifi', fig_folder, 'high')


ifi_low12 = ifi_low[12::24]
ifi_high12 = filtro.filtra_dados(ifi_low12, reference.index[12::24], 'ifi', fig_folder, 'high')

ifi_low15 = ifi_low[15::24]
ifi_high15 = filtro.filtra_dados(ifi_low15, reference.index[15::24], 'ifi', fig_folder, 'high')

res = []
for i in range(0,24):
    ifi_low_c = ifi_low[i::24]
    ifi_high = filtro.filtra_dados(ifi_low_c, reference.index[15::24], 'ifi', fig_folder, 'high')
    res.append(ifi_high)


k = 0
for r in res:
    plt.plot(r, label = str(k))
    k+=1

plt.legend()



subset = reanal_subset
subset['ssh'].load()


locsubset = subset.isel(latitude=0, longitude=0)
mod_ssh = locsubset['ssh'].values
mod_time = locsubset['time'].values

# fazendo o teste do filtro:
# 1- passa baixa -> passa alta
mod_low = filtro.filtra_dados(mod_ssh, mod_time, 'mod', fig_folder, 'low', modelo = True)
mod_high = filtro.filtra_dados(mod_ssh, mod_time, 'mod', fig_folder, 'high')

# 2 - passa banda -> Passa banda n ta funcionando muito legal nao
mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'mod', fig_folder, 'band', modelo = True)

plt.plot(mod_high, label = 'composta')
plt.plot(mod_band, label = 'banda')
plt.legend()

k = 0
corsh = np.array([])
corsb = np.array([])
for r in res:
    ch = np.round(np.corrcoef(r, mod_high)[0][1], 3) *100
    cb = np.round(np.corrcoef(r, mod_band)[0][1], 3) *100

    corsh = np.append(corsh, ch)
    corsb = np.append(corsb, cb)
    # print(f'Corr {k} composto =  {ch}')
    # print(f'Corr {k} banda =  {cb}')
    # k+=1



reference = ifi[::12*24]
correlation = np.empty((len(subset.latitude), len(subset.longitude)))
# correlation = np.zeros((len(subset.latitude), len(subset.longitude)))

for i in range(len(subset.latitude)):
    if i >1:
        continue
    for j in range(len(subset.longitude)):
        model_series = subset.isel(latitude=i, longitude=j)
        correlation[i, j] = np.corrcoef(reference['ssh'], model_series['ssh'].values)[0, 1]
correlation
            # nivel_mar = filtro.filtra_dados(model_series['ssh'].values, model_series['time'].values,
            #                                 'modelo', fig_folder, 'band', True )
            # ifilt = filtro.filtra_dados(np.asarray(reference['ssh']), 
            #                             np.asarray(reference.index), 'ponto', fig_folder, 'band', True)
            # correlation[i,j] = np.corrcoef(nivel_mar, ifilt1)[0,1]
            # correlation[i,j] = np.corrcoef(np.asarray(ifisc[11::24]['257']), model_series['ssh'].values)[0, 1]

#

# teste corr hora

reference = ifi[::12*24]

correlation = np.empty((24, len(subset.latitude), len(subset.longitude)))

reference = ifi[h*12::12*24]
cors = {}
for h in range(24):
    correlation = np.zeros((len(subset.latitude), len(subset.longitude)))

    reference = ifi[h*12::12*24]

    for i in range(len(subset.latitude)):
        if i >1:
            continue
        for j in range(len(subset.longitude)):
            model_series = subset.isel(latitude=i, longitude=j)
            correlation[i, j ] = np.corrcoef(reference['ssh'], model_series['ssh'].values)[0, 1]

    cors[h] = correlation
correlation


# teste hora:

correlation = np.empty((len(subset.latitude), len(subset.longitude)))
for i in range(len(subset.latitude)):
    for j in range(len(subset.longitude)):
        model_series = subset.isel(latitude=i, longitude=j)

        nivel_mar = filtro.filtra_dados(model_series['ssh'].values, model_series['time'].values,
                                        'modelo', fig_folder, 'band', True)
        
        correlation[i, j] = np.corrcoef(resultado, nivel_mar)[0, 1]


            # ifilt = filtro.filtra_dados(np.asarray(reference['ssh']), 
            #                             np.asarray(reference.index), 'ponto', fig_folder, 'band', True)
            # correlation[i,j] = np.corrcoef(nivel_mar, ifilt1)[0,1]
            # correlation[i,j] = np.corrcoef(np.asarray(ifisc[11::24]['257']), model_series['ssh'].values)[0, 1]


result = []
for i in range(0,24,6):
    reference =  ifi[ifi.index.time == pd.to_datetime(f'{i}:00:00').time()]
    ifilt = filtro.filtra_dados(np.asarray(reference['ssh']), 
                                np.asarray(reference.index), 'ponto', fig_folder, 'band', True)
    result.append(ifilt)



interp = ifi.resample('1D').interpolate('cubic')
intepfilt = filtro.filtra_dados(np.asarray(ifi.resample('1D').interpolate()['ssh']), 
                                np.asarray(ifi.resample('1D').interpolate().index), 'ponto', fig_folder, 'band', True)
plt.plot(intepfilt[:30], label = 'interp')
s = 0
for k in result:
    plt.plot(k[:30], label = str(s))
    s+=6
plt.legend()




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
# ax1.plot(time_hycom, hy*100, label = 'HYCOM')
ax1.plot(reference.index[::24], ifi_high, label = 'Ilha Fiscal Diário')
ax1.grid()
ax1.legend(fontsize='x-small', loc='lower left')



    # p cima ok

scp bcabral@146.164.4.200:/home/bcabral/map.png /Users/breno/Documents/Mestrado/resultados/2015/fig


    # PEGANDO A GRADE
    for ponto in pontos_dado:
     # FYI esse bloco leva 31 segundos por loop (por ponto). como sao 15 pts, leva quase 8 		#   min pra terminar de rodar tudo (por modelo)
        print('comecando ' + ponto)
        results[model][ponto] = []
        i = datetime.datetime.now()
     # botar de um jeito que serie com NaN seja ignorada!
        series ={
    	'A': reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'B' : reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'C' :  reanalisys.sel(latitude=pontos_dado[ponto][0] + lat_resolution, longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time'),
    	'D' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'E' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'F' : reanalisys.sel(latitude=pontos_dado[ponto][0], longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time'),
    	'G' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1] - lon_resolution, method='nearest').dropna('time'),
    	'H' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1], method='nearest').dropna('time'),
    	'I' : reanalisys.sel(latitude=pontos_dado[ponto][0] - lat_resolution, longitude=pontos_dado[ponto][1] + lon_resolution, method='nearest').dropna('time')
    	}
    	
    	# tem que importar a itertools
        combinacoes = list(itertools.combinations(series.keys(), 2))
        for serie1, serie2 in combinacoes:
            try:
                ks_statistic, p_value = ks_2samp(series[serie1]['ssh'], series[serie2]['ssh'])
                print(f"KS Statistic between {serie1} and {serie2}: {ks_statistic}, p-value: {p_value}")
                results[model][ponto].append((serie1, serie2, ks_statistic, p_value))
            except Exception:
                print("Serie com Nan " + serie1 +  serie2)
	    	
    	# TODO: analise dos pontos
        print(f'Tempo de execução {(datetime.datetime.now() - i).seconds} segundos')
	## Ver como vou fazer pra fazer essas diferencas
        
    print('____OK____' + model)
    print(f'Tempo de execução {(datetime.datetime.now() - i).seconds} segundos')


modelos = ['BRAN', 'CGLO', 'ECCO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  ,'SODA']
for modelo in modelos:
    f'scp -r bcabral@146.164.4.200:/data3/MOVAR/modelos/REANALISES/{modelo}/SSH/2015/ /Users/breno/model_data/{modelo}/'
    scp -r bcabral@146.164.4.200:/data3/MOVAR/modelos/REANALISES/BRAN/SSH/2015/ /Users/breno/model_data/BRAN/