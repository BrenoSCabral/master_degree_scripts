# arquivo feito pra pegar os resultados de corrente pra reanalise e filtra-los
import os
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd
import datetime
import math


import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import matplotlib.dates as mdates


sys.path.append(
    '../old'
)
sys.path.append(
    '../dynamical_analysis'
)
sys.path.append(
    '../'
)
# my files
from read_reanalisys import set_reanalisys_curr_dims
import filtro
# import plot_hovmoller as ph
import model_filt
import stats
import general_plots as gplots

import matplotlib
matplotlib.use('TkAgg')

model_path = '/data3/MOVAR/modelos/REANALISES/'
# model_path = '/Users/breno/model_data/BRAN_CURR'
fig_folder = '/home/bcabral/mestrado/fig/isobaths_50/'



# models =  ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']

def get_reanalisys(lat, lon, model, di, df):
    reanal = {}
    years = list(set([di.year, df.year]))
    for year in years:
        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + '/*.nc')
                                           , model)        
        # reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
        #                                     , model)
        
    reanalisys = xr.concat(list(reanal.values()), dim="time")
    model_series = reanalisys.sel(latitude=lat, longitude=lon, depth=0, method='nearest')
    model_series = model_series.sel(time=slice(di, df))

    mod_u = model_series['u'].values
    mod_v = model_series['v'].values

    # importante fazer aqui primeiro a conversao desse cara pra direcao along shore
    mod_int = np.sqrt(mod_u **2 + mod_v **2) 
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


def curr_dir(u, v):
    return (180/np.pi) * math.atan2(u, v) # + 180 -> existe no caso de vento, que eh de onde vem


def convert_coords(u, v):
    # converter as coordenadas nao faz diferenca, fica o mesmo valor.
    # acho que o que importa mais eh corrigir a orientacao da praia
    direction = curr_dir(u,v)
    cart_direction = CalcGeographicAngle(direction)
    vel = np.sqrt(u**2 + v **2)

    converted_u = vel * np.cos(np.deg2rad(cart_direction))
    converted_v = vel * np.sin(np.deg2rad(cart_direction))
    
    converted_u = converted_u.round(1)
    if u != converted_u:
        print('DEU RUIM!U')
        print(u)

    converted_v = converted_v.round(1)
    if v != converted_v:
        print('DEv RvIM!V')
        print(v)
    # print(f'''
    # u = {u}
    # v = {v}

    # u convertido = {u}
    # v convertido = {v}
    # ''')


def rotate_current(U, V, theta_deg):

    # theta_deg_conv = theta_deg
    theta_deg_conv = CalcGeographicAngle(theta_deg) # se converte, o cross shore pos entra na costa, mas
                                                        # n muda o along shore

    theta_rad = np.deg2rad(theta_deg_conv)

    cos_theta = np.cos(theta_rad)

    sin_theta = np.sin(theta_rad)

    U_prime = cos_theta * U + sin_theta * V  # Along-shore

    V_prime = -sin_theta * U + cos_theta * V  # Cross-shore

    return U_prime, V_prime



# Ângulos de costa: 0° (N), 45° (NE), 90° (E)

# angles = range(0,360,5)

# results = {angle: rotate_current(U, V, angle) for angle in angles}

# cross = []
# along = []
# for i in results.keys():
#     cross.append(results[i][0])
#     along.append(results[i][1])


# plt.plot(angles, cross)
# plt.grid()
# plt.show()

''' 
<TODO>
1 OK - Definir as 7 linhas que vou pegar as secoes transversais
1.1 OK - Sul, Antes, Durante e Após a redução da plataforma, Bahia, Sergipe, Fortaleza
2 - OK Definir um quadrilatero com esta linha como diagonal e pegar a corrente dentro deste quadrilatero
3 - +-OK interpolar linearmente este dado de corrente para obter uma linha mais "reta"
4 - OK selecionar os pontos que ficam em cima da linha definida
5 - Comparar como ficam os resultados se eu filtrar antes ou depois de compor a direcao along-shore
6 - comparar os resultados do along-shore filtrado com o nao filtrado e observar como a passagem da OCC influencia a CB
7 - Observar o decaimento desta influencia cross-shore
'''

# year = 2015
model = 'BRAN'


# reanal = xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
# print('______________________________')
# print(model, list(reanal.indexes))# list(reanal.keys()))


reanal = {}
years = [2015]
for year in years:
    reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                        , model)
    
reanalisys = xr.concat(list(reanal.values()), dim="time")

# define a linha que eu vou plotar
pts_cross = get_cross_points()
sections = []
for i in range(len(pts_cross)):
    if i%2 != 0:
        continue
    start = (pts_cross.iloc[i]['lon'], pts_cross.iloc[i]['lat'])
    end = (pts_cross.iloc[i + 1]['lon'], pts_cross.iloc[i + 1]['lat'])

    cross_line = interpolate_points(start, end, 10)

    sections.append(cross_line)

num_sec = 0
for section in sections:
    num_sec += 1
    print(f'iniciando secao {num_sec}')
    # section = sections[0]

    lat_i = section['lat'].min() - 0.3
    lat_f = section['lat'].max() + 0.3
    lon_i = section['lon'].min() - 0.3
    lon_f = section['lon'].max() + 0.3

    # preciso pegar agora o quadrilatero em torno da linha pra fazer a interpolacao
    reanal_subset = reanalisys.where((reanalisys.latitude < lat_f) & 
                                (reanalisys.longitude < lon_f) &
                                (reanalisys.latitude > lat_i) & 
                                (reanalisys.longitude > lon_i) ,
                                drop=True)

    # dependendo do modelo que for usar, acho que nao precisa nem interpolar. Se for necessario, achei essa resposta
    # -> https://stackoverflow.com/questions/73455966/regrid-xarray-dataset-to-a-grid-that-is-2-5-times-coarser



    # Supondo que reanal_subset e section já estão carregados

    # Inicializa listas para armazenar os dados
    section_u = []
    section_v = []

    # Extrair os dados para cada ponto na linha definida por section
    for index, row in section.iterrows():
        lon = row['lon']
        lat = row['lat']
        
        # Seleciona os dados em função da longitude e latitude, e calcula a média ao longo do tempo
        section_u.append(reanal_subset['u'].sel(longitude=lon, latitude=lat, method='nearest').mean(dim='time').values)
        section_v.append(reanal_subset['v'].sel(longitude=lon, latitude=lat, method='nearest').mean(dim='time').values)


    # Converter para arrays numpy
    section_u = np.array(section_u)
    section_v = np.array(section_v)

    # # Obter os valores de profundidade
    depths = reanal_subset['depth'].values


    # # Criar um gráfico de contorno para v
    # plt.figure(figsize=(10, 6))

    # # Plotar a seção de v
    # plt.contourf(section['lon'], depths, section_v.T, levels=50, cmap='viridis')  # Transpondo para profundidade vs longitude
    # plt.colorbar(label='v (m/s)')
    # plt.title('Seção de v ao longo da linha definida por section (média ao longo do tempo)')
    # plt.xlabel('Longitude')
    # plt.ylabel('Profundidade (m)')
    # plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
    # plt.tight_layout()
    # plt.show()





    delta_lon = section['lon'].values[-1] - section['lon'].values[0]
    delta_lat = section['lat'].values[-1] - section['lat'].values[0]
    theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
    theta_deg = np.degrees(theta_rad)  # Convertendo para graus

    # theta_geo = CalcGeographicAngle(theta_deg)
    # theta_geo_rad = np.radians(theta_geo)  # Converter de volta para radianos

    # ou compoe a intensidade e multiplica pelo cos do angulo
    # section_int = np.sqrt(section_u **2 + section_v**2)
    # section_int_rotated = section_int * np.cos(theta_rad)


    # u_rotated = section_u * np.cos(theta_adj) + section_v * np.sin(theta_adj)
    # v_rotated = -section_u * np.sin(theta_adj) + section_v * np.cos(theta_adj)

    along_shore, cross_shore = rotate_current(section_u, section_v, theta_deg)

    plt.figure(figsize=(10, 6))

    # Plotar a seção
    plt.contourf(section['lon'], depths, along_shore.T, levels=50, cmap='bwr')  # Transpondo para profundidade vs longitude
    plt.colorbar(label='Int. (m/s)')
    plt.title('Média de velocidade normal à seção')
    plt.xlabel('Longitude')
    plt.ylabel('Profundidade (m)')
    plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
    plt.tight_layout()
    plt.savefig(f'/home/bcabral/mestrado/fig/curr_section_raw/{model}_{num_sec}')
    plt.close()


# fazendo a mesma analise so que agora filtrado:

num_sec = 0
for section in sections:
    num_sec += 1
    print(f'iniciando secao {num_sec}')
    # section = sections[0]

    lat_i = section['lat'].min() - 0.2
    lat_f = section['lat'].max() + 0.2
    lon_i = section['lon'].min() - 0.2
    lon_f = section['lon'].max() + 0.2

    # preciso pegar agora o quadrilatero em torno da linha pra fazer a interpolacao -> isso se tornou obsoleto
    reanal_subset = reanalisys.where((reanalisys.latitude < lat_f) & 
                                (reanalisys.longitude < lon_f) &
                                (reanalisys.latitude > lat_i) & 
                                (reanalisys.longitude > lon_i) ,
                                drop=True)
    
    u_filt = model_filt.filtra_reanalise_u(reanal_subset)
    v_filt = model_filt.filtra_reanalise_v(reanal_subset)

    # dependendo do modelo que for usar, acho que nao precisa nem interpolar. Se for necessario, achei essa resposta
    # -> https://stackoverflow.com/questions/73455966/regrid-xarray-dataset-to-a-grid-that-is-2-5-times-coarser



    # Supondo que reanal_subset e section já estão carregados

    # Inicializa listas para armazenar os dados
    section_u = []
    section_v = []

    # Extrair os dados para cada ponto na linha definida por section
    for index, row in section.iterrows():
        lon = row['lon']
        lat = row['lat']
        
        # Seleciona os dados em função da longitude e latitude, e calcula a média ao longo do tempo
        section_u.append(u_filt.sel(longitude=lon, latitude=lat, method='nearest').mean(dim='time').values)
        section_v.append(v_filt.sel(longitude=lon, latitude=lat, method='nearest').mean(dim='time').values)


    # Converter para arrays numpy
    section_u = np.array(section_u)
    section_v = np.array(section_v)

    # # Obter os valores de profundidade
    # isso aqui deve ser movido pra fora do loop, essa variavel nao muda
    depths = reanal_subset['depth'].values

    delta_lon = section['lon'].values[-1] - section['lon'].values[0]
    delta_lat = section['lat'].values[-1] - section['lat'].values[0]
    theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
    theta_deg = np.degrees(theta_rad)  # Convertendo para graus

    # NOVA TENTATIVA DE PEGAR A COMPONENTE ALONG-SHORE:
    ### ROTACAO DE COORDENADAS:

    # acho que pra fazer esse calculo, eu preciso usar as componentes na orientacao
    # norte, nao em cartesiano



    # Pra esse calculo dar certo, as componentes da corrente precisam ser traduzidas pro cartesiano

    # x' = x cos(a) + y sen(a)
    # y' = ycos(a) - x sen(a)



    # theta_geo = CalcGeographicAngle(theta_deg)
    # theta_geo_rad = np.radians(theta_geo)  # Converter de volta para radianos

    # ou compoe a intensidade e multiplica pelo cos do angulo
    # section_int = np.sqrt(section_u **2 + section_v**2)
    # section_int_rotated = section_int * np.cos(theta_rad)

    along_shore, cross_shore = rotate_current(section_u, section_v, theta_deg)


    # u_rotated = section_u * np.cos(theta_adj) + section_v * np.sin(theta_adj)
    # v_rotated = -section_u * np.sin(theta_adj) + section_v * np.cos(theta_adj)

    plt.figure(figsize=(10, 6))

    # Plotar a seção
    plt.contourf(section['lon'], depths, along_shore.T, levels=50, cmap='bwr')  # Transpondo para profundidade vs longitude
    plt.colorbar(label='Int. (m/s)')
    plt.title('Média de velocidade normal à seção')
    plt.xlabel('Longitude')
    plt.ylabel('Profundidade (m)')
    plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
    plt.tight_layout()
    plt.savefig(f'/home/bcabral/mestrado/fig/curr_section_filt/{model}_{num_sec}')
    plt.close()
