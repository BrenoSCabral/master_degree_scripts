import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from plota_dados_marinha import get_lat_lon, nome_estacao

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from sys import path
from crosspecs import crospecs


path.append(
    '/Users/breno/Documents/Mestrado/tese/scripts'
)
import le_dado
from filtro import filtra_dados


# files_marinha_broad = [ 
#     '30810007452505202024112020ALT.txt',
#     '30685007472206202021122020ALT.txt',
#     '30149006350202201430012015ALT.txt',
#     '30340001331804200831122015ALT.txt',
#     '10566004530107201030062011ALT.txt',
#     '10615005140101200731122008ALT.txt',
#     '40118008001204202111042022ALT.txt',
#     '10656006750101201917122019ALT.txt',
#     '40141006051510200409092005ALT.txt',
#     '40140001941903201518052016ALT.txt',
#     '10681004830107200830062010ALT.txt',
#     '10615004090812200704032009ALT.txt',
#     '40141006051302200622082006ALT.txt',
#     '30156007692010202015062021ALT.txt',
#     '40141006052510200601042008ALT.txt',
#     '40141006050311201031122020ALT.txt',
#     '40140001940101202031122020ALT.txt',
#     '30955008291912202220062023ALT.txt',
#     '10656006761411201716122019ALT.txt',
#     '30340005411804200818042010ALT.txt']

files_marinha = [
    # '30340001331804200831122015ALT.txt',
    # '10615005140101200731122008ALT.txt',
    '40140001941903201518052016ALT.txt',
    # '10681004830107200830062010ALT.txt',
    # '10615004090812200704032009ALT.txt',
    # '40141006052510200601042008ALT.txt',
    # '40141006050311201031122020ALT.txt',
    # '40140001940101202031122020ALT.txt',
    '10656006761411201716122019ALT.txt',
    # '30340005411804200818042010ALT.txt'
]

files_gloss = [
    'Fortaleza.csv',
    # 'Salvador_2004_2015.csv',
    'salvador2.csv',
    # 'salvador.csv',
    # 'Salvador_glossbrasil.csv',
    # 'Santana.csv'
]

def checa_janelas(df, point):
    # Nome do arquivo de log
    log_filename = f'/Users/breno/Documents/Mestrado/dados/nne/log_nan_{point}.txt'

    # Criar janelas interrompidas por NaN
    windows = []
    current_window = []

    for index, row in df.iterrows():
        if any(row.isnull()):  # Verifica se a linha contém pelo menos um NaN
            if current_window:  # Se a janela atual não estiver vazia, adiciona à lista de janelas
                windows.append(current_window)
                current_window = []  # Reinicia a janela atual
        else:
            current_window.append((index, row))  # Adiciona a linha à janela atual

    # Adicionar a última janela à lista de janelas, se não estiver vazia
    if current_window:
        windows.append(current_window)
            

    # Abre o arquivo de log para escrita
    with open(log_filename, 'w') as log_file:
        log_file.write(f"{point}\n")
        log_file.write("Janelas interrompidas por NaN:\n")
        for i, window in enumerate(windows, 1):
            log_file.write(f"Janela {i}:\n")
            start_time = window[0][0].strftime("%Y-%m-%d %H:%M:%S")
            end_time = window[-1][0].strftime("%Y-%m-%d %H:%M:%S")
            log_file.write(f"{start_time} - {end_time}\n")
            log_file.write("\n")  # Adiciona uma linha em branco entre as janelas

    print(f"Saída gravada em '{log_filename}'")

def read_marinha(name):
    path = f'/Users/breno/Downloads/Dados/{name}'
    
    df = pd.read_csv(path, skiprows=11, sep=';',encoding='latin-1')
    df.loc[len(df)] = df.columns.to_list()
    df.columns = ['data', 'ssh', 'Unnamed: 2']
    df = df.drop(columns=['Unnamed: 2'])

    # Converter 'data' para formato datetime, se necessário
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M')
    df.set_index('data', inplace=True)

    df['ssh'] = df['ssh'].astype(int)

    df = df.sort_index()

    lat_lon = get_lat_lon(name)
    df['lat'] = lat_lon[0]
    df['lon'] = lat_lon[1]


    return df

def find_point(name):
    pontos = {
        'cananeia' : [-25.02, -47.93],
        'fortaleza' : [-3.72, -38.47],
        'ilha_fiscal' : [-22.90, -43.17],
        'imbituba' : [-28.13, -48.40],
        'macae' : [-22.23, -41.47],
        'rio_grande' : [-32.13, -52.10],
        'salvador' : [-12.97, -38.52],
        'santana' : [-0.06, -51.17],
        #'São Pedro e São Paulo' : [3.83, -32.40],
        'ubatuba' : [-23.50, -45.12],
        'rio_grande2' : [-32.17, -52.09], #RS
        'tramandai' : [-30.00, -50.13], #RS
        'paranagua' : [-25.50, -48.53], #PR
        'pontal_sul' : [-25.55, -48.37], #PR
        'ilhabela' : [-23.77, -45.35], #SP
        'dhn' : [-22.88, -43.13],#RJ
        'ribamar': [-2.56, -44.05]#MA
        }
    
    for ponto in pontos:
        if ponto.lower() in name.lower():
            return pontos[ponto]

def read_gloss(name):
    path = f'/Volumes/BRENO_HD/GLOSS/{name}'
    ponto = find_point(name)
    serie = pd.read_csv(path)

    if len(serie.columns) == 5:
        serie.loc[len(serie)] = serie.columns.to_list()
        serie.columns=['ano', 'mes', 'dia', 'hora', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:00:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]
        
    elif len(serie.columns) == 6:
        serie.columns=['ano', 'mes', 'dia', 'hora', 'minuto', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            minu = serie ['minuto'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora', 'minuto'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]

    elif len(serie.columns) == 7:
        # serie.loc[len(serie)] = serie.columns.to_list()
        serie.columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['yyyy'][i]
            mes = serie[' mm'][i]
            dia = serie[' dd'][i]
            hora = serie [' hour'][i]
            minu = serie[' min'][i]
            seg = serie[' seg'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:{seg}')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg'], inplace=True)


        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]


    serie['ssh'] = serie['ssh'].astype(float)/10
    serie = serie.mask(serie < -1000)
    return serie

def plota_pontos(pts):
    coord = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    ax.set_extent([-55, -30, 5, -30], crs=ccrs.PlateCarree())

    # ax.set_extent([-40, 10, -50, -30], crs=coord)

    for point in pts:
        plt.plot(point[1], point[0],
                color='red', linewidth=2, marker='P',
                transform=ccrs.PlateCarree()
                )  
    lons = np.arange(-70, -20, 5)
    lats = np.arange(-25, 10, 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    plt.tight_layout()
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/nne/mapa_estacoes.png')


files = {}

for j in files_marinha:
    files[nome_estacao(j)] = read_marinha(j)
    

for i in files_gloss:
    files[i] = read_gloss(i)

# importando o simcosta:

path = '/Users/breno/Documents/Mestrado/dados/simcosta/suape.csv'
df = pd.read_csv(path, skiprows=16).dropna()
df.index = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']])
df.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'], inplace=True)
df = df.resample('H').mean()
df.columns = ['ssh']
df['lat'] = -8.393
df['lon'] = -34.96

files['suape'] = df

##############


data = files

for point in data:
    checa_janelas(data[point], point)

data.pop('suape')
data.pop('PORTO DE SALVADOR\n')


pts = []
for i in data:
    pts.append([data[i]['lat'].values[0], data[i]['lon'].values[0]])

plota_pontos(pts)
   


for i in data:
    plt.plot(data[i].index, data[i]['ssh'], label=i)
    plt.legend()
    plt.grid()
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/nne/{i}.png')
    plt.close()


# vai ter que ser isso mesmo!
# os dados da marinha estao com a altura em cm. Deixar tudo na mesma unidade
## filtrar esses dados e ver se algum sinal tipo OCC aparece aqui!
# le_dado.reamostra_dado(data['suape']['ssh'], data['suape'].index)


# vendo aqui a continuidade dos meus dados

################### vou usar so os 3 primeiros pontos
# periodo de interesse vai ser de 2017-11-14 a 2019-06-30 
# deixei o periodo um pouco mais de um ano, mas tem um problema q sao as 18 h sem dados no gloss. ver como passar por isso
import xarray as xr


rdata = {}
for i in data:
    cut_data = data[i].loc['2017-11-14':'2019-06-30']
    
    r_serie = le_dado.reamostra_dado(cut_data['ssh'], cut_data.index)

    mask = np.isnan(r_serie)
    r_serie[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r_serie[~mask])

    rdata[i] = r_serie

# agora aqui fazer a filtragem para pegar as occs. cruzar os dedos hein!
# precisa revisar a forma que eu fiz o filtro, nao ta dando certo aki

'''testeeeee'''

dia_i = '2017-12-1'
dia_f = '2018-12-31'

## fortaleza
r_serie = data['Fortaleza.csv']['ssh'].loc[dia_i:dia_f]
mask = np.isnan(r_serie)
r_serie[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r_serie[~mask])

dado_ssh = r_serie

dado_tempo = data['Fortaleza.csv']['ssh'].loc[dia_i:dia_f].index
dado_filtrado = filtra_dados(dado_ssh, dado_tempo, 'foteste', '/Users/breno/Documents/Mestrado/dados/nne', 'low',False)
dado_filtrado = dado_filtrado[744:]
dado_tempo = dado_tempo[744:]
filtro_reamostrado = le_dado.reamostra_dado(dado_filtrado, dado_tempo)
t_f = pd.date_range(start=dado_tempo[0], end=dado_tempo[-1], freq='D')
fort_filtrado = filtra_dados(filtro_reamostrado, t_f, 'foteste', '/Users/breno/Documents/Mestrado/dados/nne', 'high')

#### salvador

r_serie = data['salvador2.csv']['ssh'].loc[dia_i:dia_f]
mask = np.isnan(r_serie)
r_serie[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r_serie[~mask])

dado_ssh = r_serie

dado_tempo = data['salvador2.csv']['ssh'].loc[dia_i:dia_f].index
dado_filtrado = filtra_dados(dado_ssh, dado_tempo, 'salteste', '/Users/breno/Documents/Mestrado/dados/nne', 'low',False)
dado_filtrado = dado_filtrado[744:]
dado_tempo = dado_tempo[744:]
filtro_reamostrado = le_dado.reamostra_dado(dado_filtrado, dado_tempo)
t_f = pd.date_range(start=dado_tempo[0], end=dado_tempo[-1], freq='D')
sal_filtrado= filtra_dados(filtro_reamostrado, t_f, 'salteste', '/Users/breno/Documents/Mestrado/dados/nne', 'high')


fig = plt.figure(figsize=(14,4))

ax1= fig.add_subplot(211)
ax2= fig.add_subplot(212)

ax1.plot(t_f, fort_filtrado, label='fortaleza')
ax1.set_ylim([-10, 10])
ax1.grid()
ax1.legend()
# ax1.legend(fontsize='x-small', loc='lower left')


ax2.plot(t_f, sal_filtrado, label='salvador')
ax2.grid()
ax2.set_ylim([-10, 10])
ax2.legend()


x = sal_filtrado
y = fort_filtrado

# vendo abaixo a correlacao e o lag associado a ela
# correlacao obtida foi de cerca de 31,14% e o lag foi de 0

from scipy import signal

correlation = signal.correlate(x[80:100], y[80:100] , mode="full")
lags = signal.correlation_lags(len(x[80:100]), len(y[80:100]), mode="full")
lag = lags[np.argmax(abs(correlation))]
lag


correlation = signal.correlate(x, y , mode="full")
lags = signal.correlation_lags(len(x), len(y), mode="full")
lag = lags[np.argmax(abs(correlation))]

# plot conjunto:
m = 100

plt.plot(x[80:m], label='salvador')
plt.plot(y[80:m], label='fortaleza')
plt.grid()
plt.legend()


###### pegando modelo

path_reanalise = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/'
figs_folder = '/Users/breno/Documents/Mestrado/dados/nne/filtro'

model_name = 'GLOR12'
point_name = 'Salvador'


def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude', 'surf_el':'ssh'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude', 'Time':'time', 'eta_t':'ssh'})
    else:
    	for i in list(reanalisys.variables):
            default = ['latitude', 'longitude', 'time']
            if i not in default:
                reanalisys = reanalisys.rename({i:'ssh'})

    return reanalisys

#import filtra_dados

reanalisys = xr.open_mfdataset(path_reanalise + 'GLOR12/2018/*.nc')
reanalisys = set_reanalisys_dims(reanalisys, 'GLOR12')

glor12 = reanalisys.sel(latitude = -13.08, longitude = -38.38, method = 'nearest')['ssh']
ssh_glor12_sa = glor12.values
time_glor12 = glor12.time.values
tg12 = time_glor12 - np.timedelta64(12,'h')

glor12_filt_low = filtra_dados(ssh_glor12_sa, time_glor12, 'glor12_low', figs_folder,
                                 'low', True)

g12_sa = filtra_dados(glor12_filt_low, time_glor12, 'glor12_high', figs_folder,
                                 'high', True)

glor12 = reanalisys.sel(latitude = -3-66, longitude = -38.50, method = 'nearest')['ssh']
ssh_glor12_fo = glor12.values
time_glor12 = glor12.time.values
tg12 = time_glor12 - np.timedelta64(12,'h')

glor12_filt_low = filtra_dados(ssh_glor12_fo, time_glor12, 'glor12_low', figs_folder,
                                 'low', True)

g12_fo = filtra_dados(glor12_filt_low, time_glor12, 'glor12_high', figs_folder,
                                 'high', True)

######## HYCOM

reanalisys = xr.open_mfdataset(path_reanalise + 'HYCOM/2018/*.nc')
reanalisys = set_reanalisys_dims(reanalisys, 'HYCOM')

hycom = reanalisys.sel(latitude = -13.12, longitude = -38.48, method = 'nearest')['ssh']
ssh_hycom_sa = hycom.values
time_hycom = hycom.time.values
thy = time_hycom - np.timedelta64(12,'h')

hycom_filt_low = filtra_dados(ssh_hycom_sa, time_hycom, 'hycom_low', figs_folder,
                                 'low', True)

hy_sa = filtra_dados(hycom_filt_low, time_hycom, 'hycom_high', figs_folder,
                                 'high', True)

hycom = reanalisys.sel(latitude = -3-68, longitude = -38.48, method = 'nearest')['ssh']
ssh_hycom_fo = hycom.values
time_hycom = hycom.time.values
thy = time_hycom - np.timedelta64(12,'h')

hycom_filt_low = filtra_dados(ssh_hycom_fo, time_hycom, 'hycom_low', figs_folder,
                                 'low', True)

hy_fo = filtra_dados(hycom_filt_low, time_hycom, 'hycom_high', figs_folder,
                                 'high', True)


print(f'corr modelos salvador {np.corrcoef(hy_sa, g12_sa)[1][0]}')

print(f'corr modelos fortaleza {np.corrcoef(hy_fo, g12_fo)[1][0]}')

print(f'corr g12 salvador {np.corrcoef(x, g12_sa)[1][0]}')
print(f'corr hy salvador {np.corrcoef(x, hy_sa)[1][0]}')

print(f'corr g12 fortaleza {np.corrcoef(y, g12_fo)[1][0]}')
print(f'corr hy fortaleza {np.corrcoef(y, hy_fo)[1][0]}')



# corr cruzada

# serie reamostrada
r_serie = data['salvador2.csv']['ssh'].loc['2018-01-01':'2018-12-31']
mask = np.isnan(r_serie)
r_serie[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r_serie[~mask])
salvador_reamostrado = np.asarray(r_serie[r_serie.index.hour==0])

xx1= salvador_reamostrado
xx2= ssh_glor12_sa
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffgsa,coefgsa,confgsa,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

xx1= salvador_reamostrado
xx2= ssh_hycom_sa
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffhsa,coefhsa,confhsa,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

r_serie = data['Fortaleza.csv']['ssh'].loc['2018-01-01':'2018-12-31']
mask = np.isnan(r_serie)
r_serie[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), r_serie[~mask])
fortaleza_reamostrado = np.asarray(r_serie[r_serie.index.hour==0])

xx1= fortaleza_reamostrado
xx2= ssh_glor12_fo
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffgfo,coefgfo,confgfo,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)


xx1= fortaleza_reamostrado
xx2= ssh_hycom_fo
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffhfo,coefhfo,confhfo,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(1./fffgsa/24,coefgsa,'b')
ax1.plot(1./fffgsa/24,confgsa,'--k')
ax1.grid()
ax1.set_xlim([0,40])
ax1.set_title('Salvador vs G12', fontsize='24')

ax2.plot(1./fffhsa/24,coefhsa,'b')
ax2.plot(1./fffhsa/24,confhsa,'--k')
ax2.grid()
ax2.set_xlim([0,40])
ax2.set_title('Salvador vs HY', fontsize='24')

ax3.plot(1./fffgfo/24,coefgfo,'b')
ax3.plot(1./fffgfo/24,confgfo,'--k')
ax3.grid()
ax3.set_xlim([0,40])
ax3.set_title('Fortaleza vs G12', fontsize='24')

ax4.plot(1./fffhfo/24,coefhfo,'b')
ax4.plot(1./fffhfo/24,confhfo,'--k')
ax4.grid()
ax4.set_xlim([0,40])
ax4.set_title('Fortaleza vs HY', fontsize='24')


# filtrado



xx1= x
xx2= g12_sa
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffgsa,coefgsa,confgsa,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

xx1= x
xx2= hy_sa
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffhsa,coefhsa,confhsa,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

xx1= y
xx2= g12_fo
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffgfo,coefgfo,confgfo,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)


xx1= y
xx2= hy_fo
ppp=len(xx1)
dt=24 # diario
win=2
smo=999
ci=99
h1,h2,fffhfo,coefhfo,confhfo,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax1.plot(1./fffgsa/24,coefgsa,'b')
ax1.plot(1./fffgsa/24,confgsa,'--k')
ax1.grid()
ax1.set_xlim([0,40])
ax1.set_title('Salvador vs G12', fontsize='24')

ax2.plot(1./fffhsa/24,coefhsa,'b')
ax2.plot(1./fffhsa/24,confhsa,'--k')
ax2.grid()
ax2.set_xlim([0,40])
ax2.set_title('Salvador vs HY', fontsize='24')

ax3.plot(1./fffgfo/24,coefgfo,'b')
ax3.plot(1./fffgfo/24,confgfo,'--k')
ax3.grid()
ax3.set_xlim([0,40])
ax3.set_title('Fortaleza vs G12', fontsize='24')

ax4.plot(1./fffhfo/24,coefhfo,'b')
ax4.plot(1./fffhfo/24,confhfo,'--k')
ax4.grid()
ax4.set_xlim([0,40])
ax4.set_title('Fortaleza vs HY', fontsize='24')
##################
        # dado_ssh = ifi.sea_level
        # dado_tempo = ifi.time
        # t_f = pd.date_range(start='01-01-2014', end='31-12-2014', freq='D')
        # dado_filtrado = filtra_dados(dado_ssh, dado_tempo, 'foteste', '/Users/breno/Documents/Mestrado/dados/nne', 'low')
        # filtro_reamostrado = le_dado.reamostra_dado(dado_filtrado, dado_tempo)
        # dado_filtrado2 = filtra_dados(filtro_reamostrado, t_f,
        #                               'foteste', '/Users/breno/Documents/Mestrado/dados/nne', 'high')