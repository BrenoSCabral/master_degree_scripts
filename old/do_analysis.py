__doc__ = '''
TODO: create doc

Created by Breno S. Cabral, Jan 2024
'''



''' 
TODO: Fix recursive creation of folders when plotting
TODO: Rever a parte da filtragem, verificar se a composta é de fato melhor
'''


# defining variables

year = 2014
data_name = 'Ilha Fiscal'
# data_lat = -22.90 ## LAT REAL
# data_lon = -43.17

# CHECAR PONTO QUE FIQUE MAIS LONGE DA COSTA

data_lat = -23.04 ## USA ESSE AQUI PRA FUNCIONAR
data_lon = -43.12

# defining paths

image_path = '/Users/breno/Documents/Mestrado/tese/figs/rep3/'
reanalise_path = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/'

data_file = 'Ilha Fiscal 2014.txt'

# import custom libraries

import le_dado
import le_reanalise
import filtro
import stats
import transection as transec

# importing generic libraries

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr


# import and treat data - already apply the filter
data_path = '/Volumes/BRENO_HD/dados_mestrado/dados/'
formato = 'csv'
# data = le_dado.roda_analise(data_file, data_path, formato=formato, nome=data_name,
#                            metodo='composto', fig_folder=image_path)

data_raw, time_raw = le_dado.roda_analise(data_file, data_path, formato=formato, nome=data_name, fig_folder=image_path)
dado_filtrado = filtro.filtra_dados(data_raw, time_raw, data_name, image_path, 'low')
filtro_reamostrado = le_dado.reamostra_dado(dado_filtrado, time_raw)
time = filtro_reamostrado.time
data = filtro.filtra_dados(filtro_reamostrado, time, data_name, image_path, 'high')

# importing and filtering reanalysis

curr_spaos = {
    'GLOR4': reanalise_path + 'GLOR4/',
    'HYCOM': reanalise_path + 'HYCOM/',
    'GLOR12': reanalise_path + 'GLOR12/',
    'BRAN': reanalise_path + 'BRAN/'
}

for spao in curr_spaos:
    curr_spaos[spao] = le_reanalise.read_reanalisys_curr(year, curr_spaos[spao])

spaos={
    'GLOR4': reanalise_path + 'GLOR4/',
    'HYCOM': reanalise_path + 'HYCOM/',
    'GLOR12': reanalise_path + 'GLOR12/',
    'BRAN': reanalise_path + 'BRAN/'
}

results = {}

for spao in spaos:
    raw_output = le_reanalise.roda_tudo(year, data_lat, data_lon,
                                                spao, data_name,
                                                spaos[spao], image_path)
    time_variable = le_reanalise.get_reanalisys_dims(spao)['time']
    low_output = filtro.filtra_dados(raw_output.data, raw_output[time_variable],
                                     spao, image_path,
                                     'low', modelo=True)
    high_output = filtro.filtra_dados(low_output, raw_output[time_variable],
                                      spao, image_path, 'high')
    spaos[spao] = high_output

# Calculate statistical metrics

stat_met = {}
dep_stat_met = {}
gen_stats = {}

for spao in spaos:
    stat_met[spao] = stats.stats(spaos[spao] * 100, data)
    dep_stat_met[spao] = stats.dependent_stats(spaos[spao] * 100, data)
    gen_stats[spao] = stats.general_stats(spaos[spao] * 100)

gen_stats['Dado'] = stats.general_stats(data)

stat_df = pd.DataFrame(stat_met, index=['CORR', 'NBIAS', 'NRMSE', 'SI']).round(2)
dep_stat_df = pd.DataFrame(dep_stat_met, index=['BIAS', 'RMSE','SCRMSE']).round(2)
gen_stat_df = pd.DataFrame(gen_stats, index=['SOMA', 'MÉDIA', 'DESVPAD', 'MEDIANA', 'MÁX.', 'MÍN']).round(2)

# Plotting

# 1 - Plot Time Series

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

axes = [ax1, ax2, ax3, ax4]

time = pd.date_range(f'{year}-01-01', f'{year}-12-31', freq='D')

for spao, axe in zip(spaos, axes):
    axe.plot(time, spaos[spao]*100, label=spao)
    axe.plot(time, data, label= data_name + ' Diário')
    axe.grid()
    axe.legend(fontsize='x-small', loc='lower left')
    axe.set_ylabel(spao)

ax.set_ylabel('(cm)')
# plt.show()
plt.tight_layout()

plt.savefig(image_path + 'compare_time_series', dpi=200)

# 2 - Plot Statistical Analysis
# por enquanto abortei essa ideia de usar o histograma de taylor. A ver se a conta ta certa memso
# está no taylor new
# Falar com Thiago Pires de Paula no Teams. Ele mandou uma coisa interessante.
# testar a geocat pra fazer o diagrama - obrigado andrioni

# ref = https://geocat-viz.readthedocs.io/en/latest/user_api/generated/geocat.viz.taylor.TaylorDiagram.html#geocat.viz.taylor.TaylorDiagram.add_contours
# kw = https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contour.html
# label_demo = https://matplotlib.org/stable/gallery/images_contours_and_fields/contour_label_demo.html


# 3 - Plot Spectral Analysis
# está feito de forma bagunçada no arquivo specs_amp.py

# 4 - Spatial Analysis
# isso aqui embaixo pode acabar virando um arquivo separado também

lat_perp, lon_perp = transec.perpendicular(transec.le_batimetria(), data_lat, data_lon)
lat_perp = lat_perp[1:]
lon_perp = lon_perp[1:]


pontos = {
        'P1': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P2': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P3': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P4': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P5': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            }
        }

for lat, lon, p in zip(lat_perp, lon_perp, pontos):
    for spao in curr_spaos:
        if spao == 'GLOR12':
            pontos[p][spao] = (curr_spaos[spao].sel(latitude=lat, longitude=lon, depth = 0,
                               method='nearest')).rename({'latitude': 'lat', 'longitude': 'lon',
                                                          'vo':'v', 'uo':'u'})
        elif spao == 'GLOR4':
            pontos[p][spao] = (curr_spaos[spao].sel(latitude=lat, longitude=lon, depth = 0,
                               method='nearest')).rename({'latitude': 'lat', 'longitude': 'lon',
                                                          'vo_glor':'v', 'uo_glor':'u'})
        elif spao == 'HYCOM':
            pontos[p][spao] = (curr_spaos[spao].sel(lat=lat, lon=lon, depth = 0,
                                                     method='nearest')).rename({
                                                         'water_u': 'u', 'water_v': 'v'})
        elif spao == 'BRAN':
            pontos[p][spao] = (curr_spaos[spao].sel(yu_ocean=lat, xu_ocean=lon, st_ocean = 0,
                                                     method='nearest')).rename(
                                                         {'yu_ocean': 'lat', 'xu_ocean': 'lon'})

# calculando a porcentagem que passa "cortando" a linha
# pega a parcela normal à linha
inc = (lat_perp[-1] - lat_perp[0]) / (lon_perp[-1] - lon_perp[0])
# vetor normal a inclinacao
#inc = 0.001
inc_90 = -1 / inc
vet = np.array([1, inc_90])
vet_norm = vet / np.linalg.norm(vet)
print(vet_norm)


#####
# precisa agora filtrar esse cara e ver quantos % passa

along_shore_vel = {
        'P1': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P2': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P3': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P4': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            },
        'P5': {
            'GLOR12': [],
            'GLOR4': [],
            'HYCOM': [],
            'BRAN': []
            }
        }

for p in pontos:
    for spao in curr_spaos:
        # impressionantemente, acessar os valores ta levando bastante tempo!
        u = pontos[p][spao]['u'] # .values
        v = pontos[p][spao]['v'] # .values

        u_long = u * vet_norm[0]
        v_long = v * vet_norm[1]

        along_shore_vel[p][spao] = np.sqrt(u_long**2 + v_long**2)

# long_shore = u * vetor_normal[0] + v * vetor_normal[1]
        
# acho que da pra usar direto o specs_amp pra filtrar. -> esse cara so serve pra ver espectro, n p filtrar

## mudar aqui abaixo o modelo a se usar


time_variable = along_shore_vel['P1']['GLOR12']['time']

serie_int = along_shore_vel['P1']['GLOR12'].values

low_output = filtro.filtra_dados(serie_int, time_variable,
                                    spao, image_path,
                                    'low', modelo=True)

high_output = filtro.filtra_dados(low_output, time_variable,
                                    spao, image_path, 'high')

filint = high_output

porc = 100 * filint/serie_int

locint = serie_int - filint # local = total - occ

serint = serie_int

porc_tot = 100*filint/serie_int

porc_loc = 100*filint/locint

porc = np.array([min(porc_tot[i], porc_loc[i]) if porc_tot[i] > 0 \
                 else max(porc_tot[i], porc_loc[i]) for i in range(len(porc_tot))])

# plt.plot(porc)
# plt.ylim([-100,100])

import scipy.stats as pystats

# plt.grid()
# bins = np.array([-100, -75, -50, -25, 0, 25, 50, 75, 100, 101])
bins = np.arange(-100, 101, 25)
plt.hist(porc, bins=bins, density=False, alpha=0.6, edgecolor='b')  # density=True normaliza o histograma 
plt.title('Ocorrências por classes de % de efeito sobre a corrente')
plt.xlabel('%')
plt.ylabel('Número de Ocorrências')
plt.savefig(image_path+'hist_int_curr_p1g12.png')
plt.close('all')
