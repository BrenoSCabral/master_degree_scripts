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
# matplotlib.use('TkAgg')

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


def old_get_cross_points():
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


def get_cross_points():
    # inicia em 0, nao em -50
    pts_cross = pd.DataFrame({
        'lon': {0: -50.57,
        1: -46.635435606453804,
        2: -47.85,
        3: -44.38592561317371,
        4: -42.02,
        5: -40.773547940652485,
        6: -40.81,
        7: -36.44319569143055,
        8: -38.99,
        9: -34.490344495807314,
        10: -37.05,
        11: -33.58410829011336,
        12: -36.83,
        13: -34.77723777841317}, 
        'lat': {0: -30.87,
        1: -33.05394323166,
        2: -25.0,
        3: -27.872384662368383,
        4: -23.0,
        5: -27.32397528552412,
        6: -21.0,
        7: -22.086933823530387,
        8: -15.0,
        9: -14.940794006523781,
        10: -11.0,
        11: -13.870191572249867,
        12: -5.0,
        13: -0.995431075920779}
    })
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



def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula a distância em quilômetros entre dois pontos na superfície da Terra especificados por sua latitude e longitude.

    Parâmetros:
    lat1, lon1: Latitude e longitude do primeiro ponto
    lat2, lon2: Latitude e longitude do segundo ponto

    Retorna:
    Distância entre os dois pontos em quilômetros.
    """
    # Convertendo graus para radianos
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Diferenças das coordenadas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Fórmula do haversine
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Raio da Terra em km
    R = 6371

    # Distância
    distance = R * c
    return distance


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

# year = 2015
model = 'BRAN'
def plot_poins():
    for section in sections:
# section = sections[0]

        coast = section.iloc[0]
        openp = section.iloc[-1]

        coastal_point = reanalisys.sel(longitude=coast['lon'], latitude=coast['lat'], depth=0, method='nearest')
        open_point = reanalisys.sel(longitude=openp['lon'], latitude=openp['lat'], depth=0, method='nearest')

        u_coast = coastal_point['u']
        v_coast = coastal_point['v']

        u_open = open_point['u']
        v_open = open_point['v']

        # pegando o filtrado

        u_coast_filt = model_filt.filtra_reanalise_u(coastal_point)
        v_coast_filt = model_filt.filtra_reanalise_v(coastal_point)


        u_open_filt = model_filt.filtra_reanalise_u(open_point)
        v_open_filt = model_filt.filtra_reanalise_v(open_point)
        # rotacionando

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus

        along_shore, cross_shore = rotate_current(u_coast, v_coast, theta_deg)

        along_shore_filt, cross_shore_filt = rotate_current(u_open_filt, v_open_filt, theta_deg)

        # extrair valores pra plotar mais bonitinho

        time = along_shore['time'].values

        data = along_shore.values
        data_filt = along_shore_filt.values


        plt.figure(figsize=(15, 5))
        plt.plot(time, data, label="Sem filtro")
        plt.plot(time, data_filt, label="Filtrado")


        # Formatar o eixo de tempo
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %Y'))  # Meses e anos
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())  # Marca por mês
        plt.xticks(rotation=45)  # Rotacionar rótulos

        # Adicionar título e rótulos
        plt.title("Componente Along-Shore")
        plt.xlabel("Tempo")
        plt.ylabel("Velocidade (m/s)")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'/home/bcabral/mestrado/fig/curr_teste_comp.png')

# plotar os dados diarios e no final fazer um gif


### GARBAGE ABOVEEEE
# Função para criar gráficos


def plot_section(lons, depths, section_values, title, output_path, lvls = 50):
    fig= plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    ax.set_facecolor([0,0,0,0.6])
    plt.contourf(lons, depths, section_values.T, levels=lvls, cmap='magma')
    plt.colorbar(label='Int. (m/s)')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Profundidade (m)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def four_window_fig(top_e, top_d, bot_e, bot_d, path):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    ticks1 = np.arange(-.6, .6, .03)


    ## Esquerda superior -----------------------------------

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap='bwr', levels=ticks1)
    ax[0,0].set_facecolor([0,0,0,0.6]) 


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black', levels=ticks1, linestyles= 'dashed',linewidths=0.5 )
    ax[0,0].clabel(cs1, inline=True,fontsize=10)


    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------


    ticks2 = np.arange(-.6,.6, .05)

    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap='bwr', levels=ticks2)
    ax[0,1].set_facecolor([0,0,0,0.6]) 
    #plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
    #             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap='bwr', levels=ticks2)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    ax[1,0].set_ylabel("Depth (m)")


    ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
                fontsize=11, fontweight='bold', color='black', ha='center',va='center',
                bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))



    ## Direita Inferior -----------------------------------



    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap='bwr', levels=ticks2)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])


    plt.tight_layout()
    #plt.suptitle('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
    plt.suptitle('Reanalysis: BRAN', fontsize=10, fontweight='bold',x=0.85)

    plt.savefig(path)


def four_window_fig_filt(top_e, top_d, bot_e, bot_d, path):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    # ticks1 = np.arange(-.2, .2, .0005)


    ## Esquerda superior -----------------------------------

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap='bwr') # , levels=ticks1)
    ax[0,0].set_facecolor([0,0,0,0.6]) 


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black', linestyles= 'dashed',linewidths=0.5 ) # , levels=ticks1
    ax[0,0].clabel(cs1, inline=True,fontsize=10)


    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])

    ax[0,0].set_title('Alongshore Filtered Flux: 0 - 400m', fontsize=10, loc='left')

    ## Direita superior -----------------------------------


    ticks2 = np.arange(-.2,.2, .0005)

    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap='bwr') # , levels=ticks2)
    ax[0,1].set_facecolor([0,0,0,0.6]) 
    #plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
    #             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black', linestyles= 'dashed',linewidths=0.5 ) # levels=ticks2, 
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap='bwr') # , levels=ticks2)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors='black',linestyles= 'dashed',linewidths=0.5 ) # levels=ticks2, 
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    ax[1,0].set_ylabel("Depth (m)")


    ax[1,0].set_title('Alongshore Filtered Flux: 400 - 4500m', fontsize=10, loc='left')


    ## Direita Inferior -----------------------------------



    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap='bwr') #, levels=ticks2)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors='black', linestyles= 'dashed',linewidths=0.5 )# levels=ticks2, 
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])


    plt.tight_layout()
    #plt.suptitle('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
    plt.suptitle('Reanalysis: BRAN', fontsize=10, fontweight='bold',x=0.85)

    plt.savefig(path)


def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks, cmap='magma'):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    # dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    # dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap=cmap, levels=ticks)
    ax[0,0].set_facecolor([0,0,0,0.6]) 


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[0,0].clabel(cs1, inline=True,fontsize=10)

    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, levels=ticks)
    ax[0,1].set_facecolor([0,0,0,0.6]) 
    #plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
    #             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap=cmap, levels=ticks)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    ax[1,0].set_ylabel("Depth (m)")


    ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
                fontsize=11, fontweight='bold', color='black', ha='center',va='center',
                bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))



    ## Direita Inferior -----------------------------------



    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, levels=ticks)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])

    # Adicionar colorbar à direita
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, label='', extend='both')


    # plt.tight_layout()
    #plt.suptitle('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
    plt.suptitle('Reanalysis: BRAN', fontsize=10, fontweight='bold',x=0.85)

    plt.savefig(path)


def four_window_custom_sem_lim(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, cmap='magma'):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    # dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    # dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap=cmap)
    ax[0,0].set_facecolor([0,0,0,0.6]) 


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black', linestyles= 'dashed',linewidths=0.5)
    ax[0,0].clabel(cs1, inline=True,fontsize=10)

    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap)
    ax[0,1].set_facecolor([0,0,0,0.6]) 
    #plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
    #             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black', linestyles= 'dashed',linewidths=0.5)
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap=cmap)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors='black', linestyles= 'dashed',linewidths=0.5)
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    ax[1,0].set_ylabel("Depth (m)")


    ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
                fontsize=11, fontweight='bold', color='black', ha='center',va='center',
                bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))



    ## Direita Inferior -----------------------------------



    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors='black', linestyles= 'dashed',linewidths=0.5)
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])

    # plt.tight_layout()
    #plt.suptitle('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
    plt.suptitle('Reanalysis: BRAN', fontsize=10, fontweight='bold',x=0.85)

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    plt.savefig(path)


def curr_window(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path,ticks, cmap='cool_r', cbar_label=''):


    if cbar_label != '%':
        vals = np.array([abs(np.nanmax(top_e)), abs(np.nanmax(top_d)),
                            abs(np.nanmin(top_e)), abs(np.nanmin(top_d)),
                            abs(np.nanmax(bot_e)), abs(np.nanmax(bot_d)),
                            abs(np.nanmin(bot_e)), abs(np.nanmin(bot_d))])
        
        vals_max = np.nanmax(vals)

        i = 0.001
        round = 5
        while vals_max > i:
            i = i*10
            round-=1


        top_e = np.round(top_e, round)
        top_d = np.round(top_d, round)
        bot_e = np.round(bot_e, round)
        bot_d = np.round(bot_d, round)

        vals = np.array([abs(np.nanmax(top_e)), abs(np.nanmax(top_d)),
                            abs(np.nanmin(top_e)), abs(np.nanmin(top_d)),
                            abs(np.nanmax(bot_e)), abs(np.nanmax(bot_d)),
                            abs(np.nanmin(bot_e)), abs(np.nanmin(bot_d))])
        
        vals_max = np.round(np.nanmax(vals) + 10**-(round), round)

        while (str(vals_max * 10 **round).split('.')[0][-1] != '5' and \
            str(vals_max * 10 **round).split('.')[0][-1] != '0'):
            
            vals_max = np.round(vals_max + 10**-(round), round)
    # futura alternativa eh somar o vals_max ate chegar o multiplo de 5 mais perto e
    # fazer passo de 5 * 10**round
    # (np.round(vals_max, round) * 10**round)

    if cbar_label == '%':
        ticks = np.arange(0, 101, 5)
    elif np.isnan(vals_max):
        print(f'Nan no plot {path}')
    elif cmap != 'cool_r':
        ticks = np.round(np.arange(-vals_max, vals_max + 10**-round, 5 * 10**-round), round)
        if len(ticks) < 10:
            ticks = np.round(np.arange(-vals_max, vals_max + 10**-round, 2.5 * 10**-round), round)
    else:
        ticks = np.round(np.arange(0, vals_max + 10**-round, 5 * 10**-round), round)
        if len(ticks) < 10:
            ticks = np.round(np.arange(0, vals_max + 10**-round, 2.5 * 10**-round), round)

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    # dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    # dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap=cmap, levels=ticks)
    ax[0,0].set_facecolor([0,0,0,0.6]) 


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[0,0].clabel(cs1, inline=True,fontsize=10)

    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    # ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, levels=ticks)
    ax[0,1].set_facecolor([0,0,0,0.6]) 
    #plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
    #             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap=cmap, levels=ticks)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    # ax[1,0].set_ylabel("Depth (m)")




    ## Direita Inferior -----------------------------------



    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, levels=ticks)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors='black', linestyles= 'dashed',linewidths=0.5, levels=ticks)
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])

    # Adicionar colorbar à direita
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    fig.text(0.05, 0.5, 'Profundidade (m)', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.text(0.5, 0.05, 'Distância da costa (km)', ha='center', va='center', rotation='horizontal', fontsize=12)

    plt.savefig(path + '.png')






# reanal = xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
# print('______________________________')
# print(model, list(reanal.indexes))# list(reanal.keys()))

pts_cross = get_cross_points()
sections = []
for i in range(len(pts_cross)):
    if i%2 != 0:
        continue
    start = (pts_cross.iloc[i]['lon'], pts_cross.iloc[i]['lat'])
    end = (pts_cross.iloc[i + 1]['lon'], pts_cross.iloc[i + 1]['lat'])

    cross_line = interpolate_points(start, end, 200)

    sections.append(cross_line)

# reanal = {}
years = range(1993, 2023)

 
s = 0
for section in sections:
    s+=1
    print(f'Iniciando seção {s}')

    section = section[:-20] # corta uma parte de mar
    dist = []
    for i, lat, lon in section.itertuples():
        if i ==0:
            lat0 = lat
            lon0 = lon
            dist.append(0)
        else:
            dist.append(haversine(lat0, lon0, lat, lon))
    mean_time = []
    var_time = []
    varf_time = []
    for year in years:
        reanal = {}
        print('comecando ' + str(year))

        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                            , model)

        reanalisys = xr.concat(list(reanal.values()), dim="time")
        # year = 'total'
        # define a linha que eu vou plotar

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus


        lat_i = section['lat'].min() - 0.15
        lat_f = section['lat'].max() + 0.15
        lon_i = section['lon'].min() - 0.15
        lon_f = section['lon'].max() + 0.15

        # preciso pegar agora o quadrilatero em torno da linha pra fazer a interpolacao
        reanal_subset = reanalisys.where((reanalisys.latitude < lat_f) & 
                                    (reanalisys.longitude < lon_f) &
                                    (reanalisys.latitude > lat_i) & 
                                    (reanalisys.longitude > lon_i) &
                                    (reanalisys.depth < 1000),
                                    drop=True)

        # Aplicar a rotação a cada ponto de grade
        reanal_subset = reanal_subset.chunk(dict(time=-1))

        along_shore, cross_shore = xr.apply_ufunc(
            rotate_current,
            reanal_subset['u'],  # Entrada U
            reanal_subset['v'],  # Entrada V
            theta_deg,           # Ângulo como escalar
            input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
            output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
            vectorize=True,  # Permite a aplicação para todas as grades
            dask="parallelized",  # Habilita o processamento paralelo
            output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
        )

        # Adicionar as componentes rotacionadas ao Dataset
        along_shore = along_shore.chunk({'depth': 17, 'latitude': 28, 'longitude': 45, 'time': 30})
        print('comecou a computar os dados along_shore')
        along_shore = along_shore.compute()
        print('terminou')
        reanal_subset['along_shore'] =  along_shore.transpose("time", "depth", "latitude", "longitude")
        reanal_subset['cross_shore'] = cross_shore

        # fazendo a mesma coisa para filtrado:

        reanal_subset['along_shore_filt'] = model_filt.filtra_reanalise_along(reanal_subset)


        # pegar a variância e a média do along_shore
        mean_along = reanal_subset['along_shore'].mean(dim='time')
        var_along = reanal_subset['along_shore'].var(dim='time')

        # pegar a variância do along_shore_filt
        var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

        lats = section['lat'].values
        lons = section['lon'].values

        mean_time.append(mean_along)
        var_time.append(var_along)
        varf_time.append(var_alongf)




        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,200))
        mean_bot= mean_along.sel(depth=slice(200,1000))

        var_top= var_along.sel(depth=slice(0,200))
        var_bot= var_along.sel(depth=slice(200,1000))

        varf_top = var_alongf.sel(depth=slice(0,200))
        varf_bot = var_alongf.sel(depth=slice(200,1000))


        mean_top_e = []
        mean_top_d = []
        mean_bot_e = []
        mean_bot_d = []

        var_top_e = []
        var_top_d = []
        var_bot_e = []
        var_bot_d = []

        varf_top_e = []
        varf_top_d = []
        varf_bot_e = []
        varf_bot_d = []


        for lat, lon in zip(lats, lons):
            m_lons = int(len(lons)/4) 
            # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[m_lons]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            elif lon == lons[m_lons]:
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            else:
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/curr_0307'
        os.makedirs(output_dir + f'/{s}/{year}', exist_ok=True)
        # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
        # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


        # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


        # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
        # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

        print('comecando plot ')

        vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                            abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
        
        vals_mean_max = np.max(vals_mean)

        # depois eh necessario repensar na questao do intervalo

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir + f'/{s}/{year}/mean',
                            np.arange(-vals_mean_max, vals_mean_max+(vals_mean_max)/10,(vals_mean_max)/5), cmap='bwr', cbar_label='(m/s)')


        vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
                            abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
        
        vals_var_max = np.max(vals_var)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir + f'/{s}/{year}/var',
                        np.arange(0,vals_var_max, vals_var_max/10), cbar_label='(m/s)\u00B2')

        vals_std_max = np.sqrt(vals_var_max)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        np.sqrt(var_top_e), np.sqrt(var_top_d),
                        np.sqrt(var_bot_e), np.sqrt(var_bot_d),
                        output_dir + f'/{s}/{year}/std',
                        np.arange(0,vals_var_max, vals_var_max/10), cbar_label='(m/s)')

        vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
                            abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        vals_varf_max = np.max(vals_varf)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir + f'/{s}/{year}/varf',
                        np.arange(0,vals_varf_max, vals_varf_max/10), cbar_label='(m/s)\u00B2')

        vals_stdf_max = np.sqrt(vals_varf_max)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        np.sqrt(varf_top_e), np.sqrt(varf_top_d),
                        np.sqrt(varf_bot_e), np.sqrt(varf_bot_d),
                        output_dir + f'/{s}/{year}/stdf',
                        np.arange(0,vals_stdf_max, vals_stdf_max/10), cbar_label='(m/s)') 


        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir + f'/{s}/{year}/perc', np.arange(0,101,10), cbar_label='%')
        

        perc_top_e_st = (np.asarray(np.sqrt(varf_top_e))/np.asarray(np.sqrt(var_top_e))) * 100
        perc_top_d_st = (np.asarray(np.sqrt(varf_top_d))/np.asarray(np.sqrt(var_top_d))) * 100
        perc_bot_e_st = (np.asarray(np.sqrt(varf_bot_e))/np.asarray(np.sqrt(var_bot_e))) * 100
        perc_bot_d_st = (np.asarray(np.sqrt(varf_bot_d))/np.asarray(np.sqrt(var_bot_d))) * 100

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        perc_top_e_st, perc_top_d_st, perc_bot_e_st, perc_bot_d_st,
                        output_dir + f'/{s}/{year}/perc_st', np.arange(0,101,10),
                        cbar_label='%')
        

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                     mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean',
        #                     np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        # four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                     mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim', cmap='bwr')

        # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
        #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
        
        # vals_var_max = np.round(np.max(vals_var),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var',
        #                 np.arange(0,vals_var_max, vals_var_max/10))

        # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
        #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        # vals_varf_max = np.round(np.max(vals_varf),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_filt',
        #                 np.arange(0,vals_varf_max, vals_varf_max/10))

        # perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        # perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        # perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        # perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc', np.arange(0,101,10))
        
        # four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim')

        # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
        # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
        # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
        # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

        # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
        #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
        
        # vals_dif_max = np.round(np.nanmax(vals_dif),2)
        
        # if np.isnan(np.nan):
        #     vals_dif_max = .05

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_diff',
        #                 np.arange(0,vals_dif_max,vals_dif_max/10))
        
    # fazendo pra media do tempo:
    print('total')

    mean_time_concat = xr.concat(mean_time, dim="year")
    var_time_concat = xr.concat(var_time, dim="year")
    varf_time_concat = xr.concat(varf_time, dim="year")

    # Calcular a média ao longo dos anos
    mean_time_avg = mean_time_concat.mean(dim="year")
    var_time_avg = var_time_concat.mean(dim="year")
    varf_time_avg = varf_time_concat.mean(dim="year")

    # Selecionar a variável
    mean_top= mean_time_avg.sel(depth=slice(0,200))
    mean_bot= mean_time_avg.sel(depth=slice(200,999999))

    var_top= var_time_avg.sel(depth=slice(0,200))
    var_bot= var_time_avg.sel(depth=slice(200,999999))

    varf_top = varf_time_avg.sel(depth=slice(0,200))
    varf_bot = varf_time_avg.sel(depth=slice(200,999999))


    mean_top_e = []
    mean_top_d = []
    mean_bot_e = []
    mean_bot_d = []

    var_top_e = []
    var_top_d = []
    var_bot_e = []
    var_bot_d = []

    varf_top_e = []
    varf_top_d = []
    varf_bot_e = []
    varf_bot_d = []

    
    for lat, lon in zip(lats, lons):
        m_lons = int(len(lons)/4) 
        # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
        # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

        if lon < lons[m_lons]:
        # Extrair o perfil em profundidade para o par (lat, lon)
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        elif lon == lons[m_lons]:
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        else:
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

    output_dir = f'/home/bcabral/mestrado/fig/curr_0307'
    os.makedirs(output_dir + f'/{s}/', exist_ok=True)


    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.max(vals_mean)

    # depois eh necessario repensar na questao do intervalo

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir + f'/{s}/mean',
                        np.arange(-vals_mean_max, vals_mean_max+(vals_mean_max)/10,(vals_mean_max)/5), cmap='bwr',
                        cbar_label='(m/s)')


    vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
                        abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
    
    vals_var_max = np.max(vals_var)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir + f'/{s}/var',
                    np.arange(0,vals_var_max, vals_var_max/10), cbar_label='(m/s)\u00B2')
    
    vals_std_max = np.sqrt(vals_var_max)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    np.sqrt(var_top_e), np.sqrt(var_top_d),
                    np.sqrt(var_bot_e), np.sqrt(var_bot_d),
                    output_dir + f'/{s}/std',
                    np.arange(0,vals_std_max, vals_std_max/10), cbar_label='(m/s)')

    vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
                        abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    vals_varf_max = np.max(vals_varf)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir + f'/{s}/varf',
                    np.arange(0,vals_varf_max, vals_varf_max/10), cbar_label='(m/s)\u00B2')
    
    vals_stdf_max = np.sqrt(vals_varf_max)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    np.sqrt(varf_top_e), np.sqrt(varf_top_d),
                    np.sqrt(varf_bot_e), np.sqrt(varf_bot_d),
                    output_dir + f'/{s}/stdf',
                    np.arange(0,vals_varf_max, vals_varf_max/10), cbar_label='(m/s)')


    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir + f'/{s}/perc', np.arange(0,101,10), cbar_label='%')
    


    perc_top_e_st = (np.asarray(np.sqrt(varf_top_e))/np.asarray(np.sqrt(var_top_e))) * 100
    perc_top_d_st = (np.asarray(np.sqrt(varf_top_d))/np.asarray(np.sqrt(var_top_d))) * 100
    perc_bot_e_st = (np.asarray(np.sqrt(varf_bot_e))/np.asarray(np.sqrt(var_bot_e))) * 100
    perc_bot_d_st = (np.asarray(np.sqrt(varf_bot_d))/np.asarray(np.sqrt(var_bot_d))) * 100

    curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    perc_top_e_st, perc_top_d_st, perc_bot_e_st, perc_bot_d_st, output_dir + f'/{s}/perc_st', np.arange(0,101,10), cbar_label='%')

stop

    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_sem_lim', cmap='bwr')
    # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
    #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
    
    # vals_var_max = np.round(np.max(vals_var),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/total/var',
    #                 np.arange(0,vals_var_max, vals_var_max/10))

    # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
    #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    # vals_varf_max = np.round(np.max(vals_varf),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/total/var_filt',
    #                 np.arange(0,vals_varf_max, vals_varf_max/10))

    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc')



    # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
    # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
    # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
    # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

    # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
    #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
    
    # vals_dif_max = np.round(np.nanmax(vals_dif),2)

    # if np.isnan(np.nan):
    #     vals_dif_max = .05
        
    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/total/var_diff',
    #                 np.arange(0,vals_dif_max,vals_dif_max/10))


##############
#
#  GAMBIARRA MAXIMA PRA NAO PRECISAR ADAPTAR
#
################


s = 0
for section in sections:
    s+=1
    print(f'Iniciando seção {s}')
    mean_timea = []
    var_timea = []
    varf_timea = []

    mean_timeu = []
    var_timeu = []
    varf_timeu = []

    mean_timev = []
    var_timev = []
    varf_timev = []


    for year in years:
        reanal = {}
        print('comecando ' + str(year))

        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                            , model)

        reanalisys = xr.concat(list(reanal.values()), dim="time")
        # year = 'total'
        # define a linha que eu vou plotar

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus


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

        # rotacionar direto o reanal_subset pode ser mais facil



        # Aplicar a rotação a cada ponto de grade
        reanal_subset = reanal_subset.chunk(dict(time=-1))

        along_shore, cross_shore = xr.apply_ufunc(
            rotate_current,
            reanal_subset['u'],  # Entrada U
            reanal_subset['v'],  # Entrada V
            theta_deg,           # Ângulo como escalar
            input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
            output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
            vectorize=True,  # Permite a aplicação para todas as grades
            dask="parallelized",  # Habilita o processamento paralelo
            output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
        )

        # Adicionar as componentes rotacionadas ao Dataset
        along_shore = along_shore.chunk({'depth': 17, 'latitude': 28, 'longitude': 45, 'time': 30})
        print('comecou a computar os dados along_shore')
        along_shore = along_shore.compute()
        print('terminou')
        reanal_subset['along_shore'] =  cross_shore.transpose("time", "depth", "latitude", "longitude")
        reanal_subset['cross_shore'] = cross_shore

        # fazendo a mesma coisa para filtrado:

        reanal_subset['along_shore_filt'] = model_filt.filtra_reanalise_along(reanal_subset)
        reanal_subset['u_filt'] = model_filt.filtra_reanalise_u(reanal_subset)
        reanal_subset['v_filt'] = model_filt.filtra_reanalise_v(reanal_subset)


        # pegar a variância e a média do along_shore
        mean_along = reanal_subset['along_shore'].mean(dim='time')
        var_along = reanal_subset['along_shore'].var(dim='time')

        # pegar a variância do along_shore_filt
        var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

        mean_u = reanal_subset['u'].mean(dim='time')
        var_u = reanal_subset['u'].var(dim='time')

        # pegar a variância do u_filt
        var_uf = reanal_subset['u_filt'].var(dim='time')

        mean_v = reanal_subset['v'].mean(dim='time')
        var_v = reanal_subset['v'].var(dim='time')

        # pegar a variância do v_filt
        var_vf = reanal_subset['v_filt'].var(dim='time')

        lats = section['lat'].values
        lons = section['lon'].values

        mean_timea.append(mean_along)
        var_timea.append(var_along)
        varf_timea.append(var_alongf)

        mean_timeu.append(mean_u)
        var_timeu.append(var_u)
        varf_timeu.append(var_uf)

        mean_timev.append(mean_v)
        var_timev.append(var_v)
        varf_timev.append(var_vf)


        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,200))
        mean_bot= mean_along.sel(depth=slice(200,999999))

        var_top= var_along.sel(depth=slice(0,200))
        var_bot= var_along.sel(depth=slice(200,999999))

        varf_top = var_alongf.sel(depth=slice(0,200))
        varf_bot = var_alongf.sel(depth=slice(200,999999))


        mean_top_e = []
        mean_top_d = []
        mean_bot_e = []
        mean_bot_d = []

        var_top_e = []
        var_top_d = []
        var_bot_e = []
        var_bot_d = []

        varf_top_e = []
        varf_top_d = []
        varf_bot_e = []
        varf_bot_d = []


        mean_topu = mean_u.sel(depth=slice(0,200))
        mean_botu = mean_u.sel(depth=slice(200,999999))

        var_topu = var_u.sel(depth=slice(0,200))
        var_botu = var_u.sel(depth=slice(200,999999))

        varf_topu = var_uf.sel(depth=slice(0,200))
        varf_botu = var_uf.sel(depth=slice(200,999999))


        mean_top_eu = []
        mean_top_du = []
        mean_bot_eu = []
        mean_bot_du = []

        var_top_eu= []
        var_top_du= []
        var_bot_eu= []
        var_bot_du= []

        varf_top_eu = []
        varf_top_du = []
        varf_bot_eu = []
        varf_bot_du = []



        mean_topv= mean_v.sel(depth=slice(0,200))
        mean_botv= mean_v.sel(depth=slice(200,999999))

        var_topv= var_v.sel(depth=slice(0,200))
        var_botv= var_v.sel(depth=slice(200,999999))

        varf_topv = var_vf.sel(depth=slice(0,200))
        varf_botv = var_vf.sel(depth=slice(200,999999))


        mean_top_ev = []
        mean_top_dv = []
        mean_bot_ev = []
        mean_bot_dv = []

        var_top_ev= []
        var_top_dv= []
        var_bot_ev= []
        var_bot_dv= []

        varf_top_ev = []
        varf_top_dv = []
        varf_bot_ev = []
        varf_bot_dv = []

        
        for lat, lon in zip(lats, lons):
            m_lons = int(len(lons)/4) 
            # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[m_lons]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))


                mean_top_eu.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_eu.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_eu.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_eu.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_eu.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_eu.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))


                mean_top_ev.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_ev.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_ev.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_ev.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_ev.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_ev.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            elif lon == lons[m_lons]:
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))


                mean_top_eu.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_eu.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_eu.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_eu.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_eu.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_eu.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_du.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_du.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_du.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_du.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_du.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_du.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))

    

                mean_top_ev.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_ev.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_ev.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_ev.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_ev.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_ev.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_dv.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_dv.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_dv.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_dv.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_dv.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_dv.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            else:
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))


                mean_top_du.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_du.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_du.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_du.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_du.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_du.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))

    
                mean_top_dv.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_dv.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_dv.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_dv.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_dv.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_dv.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
        os.makedirs(output_dir + f'/curr_comp/{s}/{year}/mean', exist_ok=True)


        print('comecando plot ')

        vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                            abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
        
        vals_mean_max = np.round(np.max(vals_mean),2)

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_cross',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_cross', cmap='bwr')


        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_cross', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_cross')
        


        vals_meanu = np.array([abs(np.nanmax(mean_topu)), abs(np.nanmax(mean_botu)),
                            abs(np.nanmin(mean_topu)), abs(np.nanmin(mean_botu))])
        
        vals_mean_maxu = np.round(np.max(vals_meanu),2)

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_eu, mean_top_du, mean_bot_eu, mean_bot_du, output_dir+f'/curr_comp/{s}/{str(year)}/mean_u',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_eu, mean_top_du, mean_bot_eu, mean_bot_du, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_u', cmap='bwr')


        perc_top_eu = (np.asarray(varf_top_eu)/np.asarray(var_top_eu)) * 100
        perc_top_du = (np.asarray(varf_top_du)/np.asarray(var_top_du)) * 100
        perc_bot_eu = (np.asarray(varf_bot_eu)/np.asarray(var_bot_eu)) * 100
        perc_bot_du = (np.asarray(varf_bot_du)/np.asarray(var_bot_du)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_eu, perc_top_du, perc_bot_eu, perc_bot_du, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_cross', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_eu, perc_top_du, perc_bot_eu, perc_bot_du, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_u')
        


        vals_meanv = np.array([abs(np.nanmax(mean_topv)), abs(np.nanmax(mean_botv)),
                            abs(np.nanmin(mean_topv)), abs(np.nanmin(mean_botv))])
        
        vals_mean_maxv = np.round(np.max(vals_meanv),2)

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_ev, mean_top_dv, mean_bot_ev, mean_bot_dv, output_dir+f'/curr_comp/{s}/{str(year)}/mean_v',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_ev, mean_top_dv, mean_bot_ev, mean_bot_dv, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_v', cmap='bwr')


        perc_top_ev = (np.asarray(varf_top_ev)/np.asarray(var_top_ev)) * 100
        perc_top_dv = (np.asarray(varf_top_dv)/np.asarray(var_top_dv)) * 100
        perc_bot_ev = (np.asarray(varf_bot_ev)/np.asarray(var_bot_ev)) * 100
        perc_bot_dv = (np.asarray(varf_bot_dv)/np.asarray(var_bot_dv)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_ev, perc_top_dv, perc_bot_ev, perc_bot_dv, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_cross', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_ev, perc_top_dv, perc_bot_ev, perc_bot_dv, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_v')




        
    # fazendo pra media do tempo:
    print('total')

    mean_timea_concat = xr.concat(mean_timea, dim="year")
    var_timea_concat = xr.concat(var_timea, dim="year")
    varf_timea_concat = xr.concat(varf_timea, dim="year")

    # Calcular a média ao longo dos anos
    mean_timea_avg = mean_timea_concat.mean(dim="year")
    var_timea_avg = var_timea_concat.mean(dim="year")
    varf_timea_avg = varf_timea_concat.mean(dim="year")


    mean_timeu_concat = xr.concat(mean_timeu, dim="year")
    var_timeu_concat = xr.concat(var_timeu, dim="year")
    varf_timeu_concat = xr.concat(varf_timeu, dim="year")

    # Calcular a média ao longo dos anos
    mean_timeu_avg = mean_timeu_concat.mean(dim="year")
    var_timeu_avg = var_timeu_concat.mean(dim="year")
    varf_timeu_avg = varf_timeu_concat.mean(dim="year")


    mean_timev_concat = xr.concat(mean_timev, dim="year")
    var_timev_concat = xr.concat(var_timev, dim="year")
    varf_timev_concat = xr.concat(varf_timev, dim="year")

    # Calcular a média ao longo dos anos
    mean_timev_avg = mean_timev_concat.mean(dim="year")
    var_timev_avg = var_timev_concat.mean(dim="year")
    varf_timev_avg = varf_timev_concat.mean(dim="year")

    # Selecionar a variável
    mean_top= mean_timea_avg.sel(depth=slice(0,200))
    mean_bot= mean_timea_avg.sel(depth=slice(200,999999))

    var_top= var_timea_avg.sel(depth=slice(0,200))
    var_bot= var_timea_avg.sel(depth=slice(200,999999))

    varf_top = varf_timea_avg.sel(depth=slice(0,200))
    varf_bot = varf_timea_avg.sel(depth=slice(200,999999))


    # Selecionar a variável
    mean_topu= mean_timeu_avg.sel(depth=slice(0,200))
    mean_botu= mean_timeu_avg.sel(depth=slice(200,999999))

    var_topu= var_timeu_avg.sel(depth=slice(0,200))
    var_botu= var_timeu_avg.sel(depth=slice(200,999999))

    varf_topu= varf_timeu_avg.sel(depth=slice(0,200))
    varf_botu= varf_timeu_avg.sel(depth=slice(200,999999))

    # Selecionar a variável
    mean_topv= mean_timev_avg.sel(depth=slice(0,200))
    mean_botv= mean_timev_avg.sel(depth=slice(200,999999))

    var_topv= var_timev_avg.sel(depth=slice(0,200))
    var_botv= var_timev_avg.sel(depth=slice(200,999999))

    varf_topv= varf_timev_avg.sel(depth=slice(0,200))
    varf_botv= varf_timev_avg.sel(depth=slice(200,999999))


    mean_top_e = []
    mean_top_d = []
    mean_bot_e = []
    mean_bot_d = []

    var_top_e = []
    var_top_d = []
    var_bot_e = []
    var_bot_d = []

    varf_top_e = []
    varf_top_d = []
    varf_bot_e = []
    varf_bot_d = []


    mean_top_eu = []
    mean_top_du = []
    mean_bot_eu = []
    mean_bot_du = []

    var_top_eu = []
    var_top_du = []
    var_bot_eu = []
    var_bot_du = []

    varf_top_eu = []
    varf_top_du = []
    varf_bot_eu = []
    varf_bot_du = []

    mean_top_ev = []
    mean_top_dv = []
    mean_bot_ev = []
    mean_bot_dv = []

    var_top_ev = []
    var_top_dv = []
    var_bot_ev = []
    var_bot_dv = []

    varf_top_ev = []
    varf_top_dv = []
    varf_bot_ev = []
    varf_bot_dv = []
    
    for lat, lon in zip(lats, lons):
        m_lons = int(len(lons)/4) 
        # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
        # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

        if lon < lons[m_lons]:
        # Extrair o perfil em profundidade para o par (lat, lon)
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))


            mean_top_eu.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_eu.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_eu.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_eu.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_eu.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_eu.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))



            mean_top_ev.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_ev.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_ev.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_ev.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_ev.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_ev.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

        elif lon == lons[m_lons]:
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))


            mean_top_eu.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_eu.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_eu.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_eu.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_eu.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_eu.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_du.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_du.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_du.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_du.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_du.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_du.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))

    
            mean_top_ev.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_ev.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_ev.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_ev.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_ev.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_ev.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_dv.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_dv.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_dv.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_dv.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_dv.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_dv.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

        else:
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        
            mean_top_du.append(mean_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_du.append(mean_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_du.append(var_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_du.append(var_botu.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_du.append(varf_topu.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_du.append(varf_botu.sel(latitude=lat, longitude=lon, method='nearest'))

        
            mean_top_dv.append(mean_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_dv.append(mean_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_dv.append(var_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_dv.append(var_botv.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_dv.append(varf_topv.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_dv.append(varf_botv.sel(latitude=lat, longitude=lon, method='nearest'))

    output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
    os.makedirs(output_dir + f'/curr_comp/{s}/total/mean', exist_ok=True)



    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.round(np.max(vals_mean),2)




    vals_meanu = np.array([abs(np.nanmax(mean_topu)), abs(np.nanmax(mean_botu)),
                        abs(np.nanmin(mean_topu)), abs(np.nanmin(mean_botu))])
    
    vals_mean_maxu = np.round(np.max(vals_meanu),2)




    vals_meanv = np.array([abs(np.nanmax(mean_topv)), abs(np.nanmax(mean_botv)),
                        abs(np.nanmin(mean_topv)), abs(np.nanmin(mean_botv))])
    
    vals_mean_maxv = np.round(np.max(vals_meanv),2)

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_cross',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_cross', cmap='bwr')


    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_ev, mean_top_dv, mean_bot_ev, mean_bot_dv, output_dir+f'/curr_comp/{s}/total/mean_v',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_ev, mean_top_dv, mean_bot_ev, mean_bot_dv, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_v', cmap='bwr')
    

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_eu, mean_top_du, mean_bot_eu, mean_bot_du, output_dir+f'/curr_comp/{s}/total/mean_u',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_eu, mean_top_du, mean_bot_eu, mean_bot_du, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_u', cmap='bwr')

    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_cross', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_cross')


    perc_top_eu = (np.asarray(varf_top_eu)/np.asarray(var_top_eu)) * 100
    perc_top_du = (np.asarray(varf_top_du)/np.asarray(var_top_du)) * 100
    perc_bot_eu = (np.asarray(varf_bot_eu)/np.asarray(var_bot_eu)) * 100
    perc_bot_du = (np.asarray(varf_bot_du)/np.asarray(var_bot_du)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_eu, perc_top_du, perc_bot_eu, perc_bot_du, output_dir+f'/curr_comp/{s}/total/var_perc_u', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_eu, perc_top_du, perc_bot_eu, perc_bot_du, output_dir+f'/curr_comp/{s}/total/var_perc_u')
    


    perc_top_ev = (np.asarray(varf_top_ev)/np.asarray(var_top_ev)) * 100
    perc_top_dv = (np.asarray(varf_top_dv)/np.asarray(var_top_dv)) * 100
    perc_bot_ev = (np.asarray(varf_bot_ev)/np.asarray(var_bot_ev)) * 100
    perc_bot_dv = (np.asarray(varf_bot_dv)/np.asarray(var_bot_dv)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_ev, perc_top_dv, perc_bot_ev, perc_bot_dv, output_dir+f'/curr_comp/{s}/total/var_perc_v', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_ev, perc_top_dv, perc_bot_ev, perc_bot_dv, output_dir+f'/curr_comp/{s}/total/var_perc_v')



############ ADAPTEI
#####################


s = 0
for section in sections:
    s+=1
    print(f'Iniciando seção {s}')
    mean_time = []
    var_time = []
    varf_time = []
    for year in years:
        reanal = {}
        print('comecando ' + str(year))

        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                            , model)

        reanalisys = xr.concat(list(reanal.values()), dim="time")
        # year = 'total'
        # define a linha que eu vou plotar

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus


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

        # rotacionar direto o reanal_subset pode ser mais facil



        # Aplicar a rotação a cada ponto de grade
        reanal_subset = reanal_subset.chunk(dict(time=-1))

        along_shore, cross_shore = xr.apply_ufunc(
            rotate_current,
            reanal_subset['u'],  # Entrada U
            reanal_subset['v'],  # Entrada V
            theta_deg,           # Ângulo como escalar
            input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
            output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
            vectorize=True,  # Permite a aplicação para todas as grades
            dask="parallelized",  # Habilita o processamento paralelo
            output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
        )

        # Adicionar as componentes rotacionadas ao Dataset
        along_shore = along_shore.chunk({'depth': 17, 'latitude': 28, 'longitude': 45, 'time': 30})
        print('comecou a computar os dados along_shore')
        along_shore = along_shore.compute()
        print('terminou')
        reanal_subset['along_shore'] =  cross_shore.transpose("time", "depth", "latitude", "longitude")
        reanal_subset['cross_shore'] = cross_shore

        # fazendo a mesma coisa para filtrado:

        reanal_subset['along_shore_filt'] = model_filt.filtra_reanalise_along(reanal_subset)


        # pegar a variância e a média do along_shore
        mean_along = reanal_subset['along_shore'].mean(dim='time')
        var_along = reanal_subset['along_shore'].var(dim='time')

        # pegar a variância do along_shore_filt
        var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

        lats = section['lat'].values
        lons = section['lon'].values

        mean_time.append(mean_along)
        var_time.append(var_along)
        varf_time.append(var_alongf)




        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,200))
        mean_bot= mean_along.sel(depth=slice(200,999999))

        var_top= var_along.sel(depth=slice(0,200))
        var_bot= var_along.sel(depth=slice(200,999999))

        varf_top = var_alongf.sel(depth=slice(0,200))
        varf_bot = var_alongf.sel(depth=slice(200,999999))


        mean_top_e = []
        mean_top_d = []
        mean_bot_e = []
        mean_bot_d = []

        var_top_e = []
        var_top_d = []
        var_bot_e = []
        var_bot_d = []

        varf_top_e = []
        varf_top_d = []
        varf_bot_e = []
        varf_bot_d = []

        
        for lat, lon in zip(lats, lons):
            m_lons = int(len(lons)/4) 
            # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[m_lons]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            elif lon == lons[m_lons]:
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            else:
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
        os.makedirs(output_dir + f'/curr_comp/{s}/{year}/mean', exist_ok=True)
        # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
        # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


        # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


        # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
        # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

        print('comecando plot ')


        vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                            abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
        
        vals_mean_max = np.round(np.max(vals_mean),2)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, '/home/bcabral/mestrado/fig/along_teste_fig.png',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_cross',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_cross', cmap='bwr')

        # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
        #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
        
        # vals_var_max = np.round(np.max(vals_var),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var',
        #                 np.arange(0,vals_var_max, vals_var_max/10))

        # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
        #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        # vals_varf_max = np.round(np.max(vals_varf),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_filt',
        #                 np.arange(0,vals_varf_max, vals_varf_max/10))

        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_cross', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_cross')

        # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
        # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
        # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
        # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

        # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
        #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
        
        # vals_dif_max = np.round(np.nanmax(vals_dif),2)
        
        # if np.isnan(np.nan):
        #     vals_dif_max = .05

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_diff',
        #                 np.arange(0,vals_dif_max,vals_dif_max/10))
        
    # fazendo pra media do tempo:
    print('total')

    mean_time_concat = xr.concat(mean_time, dim="year")
    var_time_concat = xr.concat(var_time, dim="year")
    varf_time_concat = xr.concat(varf_time, dim="year")

    # Calcular a média ao longo dos anos
    mean_time_avg = mean_time_concat.mean(dim="year")
    var_time_avg = var_time_concat.mean(dim="year")
    varf_time_avg = varf_time_concat.mean(dim="year")

    # Selecionar a variável
    mean_top= mean_time_avg.sel(depth=slice(0,200))
    mean_bot= mean_time_avg.sel(depth=slice(200,999999))

    var_top= var_time_avg.sel(depth=slice(0,200))
    var_bot= var_time_avg.sel(depth=slice(200,999999))

    varf_top = varf_time_avg.sel(depth=slice(0,200))
    varf_bot = varf_time_avg.sel(depth=slice(200,999999))


    mean_top_e = []
    mean_top_d = []
    mean_bot_e = []
    mean_bot_d = []

    var_top_e = []
    var_top_d = []
    var_bot_e = []
    var_bot_d = []

    varf_top_e = []
    varf_top_d = []
    varf_bot_e = []
    varf_bot_d = []

    
    for lat, lon in zip(lats, lons):
        m_lons = int(len(lons)/4) 
        # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
        # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

        if lon < lons[m_lons]:
        # Extrair o perfil em profundidade para o par (lat, lon)
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        elif lon == lons[m_lons]:
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        else:
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

    output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
    os.makedirs(output_dir + f'/curr_comp/{s}/total/mean', exist_ok=True)
    # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
    # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


    # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))


    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.round(np.max(vals_mean),2)

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_cross',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_cross', cmap='bwr')
    # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
    #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
    
    # vals_var_max = np.round(np.max(vals_var),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/total/var',
    #                 np.arange(0,vals_var_max, vals_var_max/10))

    # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
    #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    # vals_varf_max = np.round(np.max(vals_varf),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/total/var_filt',
    #                 np.arange(0,vals_varf_max, vals_varf_max/10))

    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_cross', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_cross')

    # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
    # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
    # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
    # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

    # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
    #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
    
    # vals_dif_max = np.round(np.nanmax(vals_dif),2)

    # if np.isnan(np.nan):
    #     vals_dif_max = .05
        
    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/total/var_diff',
    #                 np.arange(0,vals_dif_max,vals_dif_max/10))



s = 0
for section in sections:
    s+=1
    print(f'Iniciando seção {s}')
    mean_time = []
    var_time = []
    varf_time = []
    for year in years:
        reanal = {}
        print('comecando ' + str(year))

        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                            , model)

        reanalisys = xr.concat(list(reanal.values()), dim="time")
        # year = 'total'
        # define a linha que eu vou plotar

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus


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

        # rotacionar direto o reanal_subset pode ser mais facil



        # Aplicar a rotação a cada ponto de grade
        reanal_subset = reanal_subset.chunk(dict(time=-1))

        # along_shore, cross_shore = xr.apply_ufunc(
        #     rotate_current,
        #     reanal_subset['u'],  # Entrada U
        #     reanal_subset['v'],  # Entrada V
        #     theta_deg,           # Ângulo como escalar
        #     input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
        #     output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
        #     vectorize=True,  # Permite a aplicação para todas as grades
        #     dask="parallelized",  # Habilita o processamento paralelo
        #     output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
        # )

        # # Adicionar as componentes rotacionadas ao Dataset
        # along_shore = along_shore.chunk({'depth': 17, 'latitude': 28, 'longitude': 45, 'time': 30})
        # print('comecou a computar os dados along_shore')
        # along_shore = along_shore.compute()
        # print('terminou')
        # reanal_subset['along_shore'] =  cross_shore.transpose("time", "depth", "latitude", "longitude")
        # reanal_subset['cross_shore'] = cross_shore

        # # fazendo a mesma coisa para filtrado:

        reanal_subset['along_shore_filt'] = model_filt.filtra_reanalise_u(reanal_subset)


        # pegar a variância e a média do along_shore
        mean_along = reanal_subset['u'].mean(dim='time')
        var_along = reanal_subset['u'].var(dim='time')

        # pegar a variância do along_shore_filt
        var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

        lats = section['lat'].values
        lons = section['lon'].values

        mean_time.append(mean_along)
        var_time.append(var_along)
        varf_time.append(var_alongf)




        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,200))
        mean_bot= mean_along.sel(depth=slice(200,999999))

        var_top= var_along.sel(depth=slice(0,200))
        var_bot= var_along.sel(depth=slice(200,999999))

        varf_top = var_alongf.sel(depth=slice(0,200))
        varf_bot = var_alongf.sel(depth=slice(200,999999))


        mean_top_e = []
        mean_top_d = []
        mean_bot_e = []
        mean_bot_d = []

        var_top_e = []
        var_top_d = []
        var_bot_e = []
        var_bot_d = []

        varf_top_e = []
        varf_top_d = []
        varf_bot_e = []
        varf_bot_d = []

        
        for lat, lon in zip(lats, lons):
            m_lons = int(len(lons)/4) 
            # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[m_lons]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            elif lon == lons[m_lons]:
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            else:
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
        os.makedirs(output_dir + f'/curr_comp/{s}/{year}/mean', exist_ok=True)
        # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
        # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


        # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


        # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
        # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

        print('comecando plot ')

        vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                            abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
        
        vals_mean_max = np.round(np.max(vals_mean),2)

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_u',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_u', cmap='bwr')

        # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
        #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
        
        # vals_var_max = np.round(np.max(vals_var),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var',
        #                 np.arange(0,vals_var_max, vals_var_max/10))

        # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
        #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        # vals_varf_max = np.round(np.max(vals_varf),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_filt',
        #                 np.arange(0,vals_varf_max, vals_varf_max/10))

        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_u', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_u')

        # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
        # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
        # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
        # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

        # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
        #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
        
        # vals_dif_max = np.round(np.nanmax(vals_dif),2)
        
        # if np.isnan(np.nan):
        #     vals_dif_max = .05

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_diff',
        #                 np.arange(0,vals_dif_max,vals_dif_max/10))
        
    # fazendo pra media do tempo:
    print('total')

    mean_time_concat = xr.concat(mean_time, dim="year")
    var_time_concat = xr.concat(var_time, dim="year")
    varf_time_concat = xr.concat(varf_time, dim="year")

    # Calcular a média ao longo dos anos
    mean_time_avg = mean_time_concat.mean(dim="year")
    var_time_avg = var_time_concat.mean(dim="year")
    varf_time_avg = varf_time_concat.mean(dim="year")

    # Selecionar a variável
    mean_top= mean_time_avg.sel(depth=slice(0,200))
    mean_bot= mean_time_avg.sel(depth=slice(200,999999))

    var_top= var_time_avg.sel(depth=slice(0,200))
    var_bot= var_time_avg.sel(depth=slice(200,999999))

    varf_top = varf_time_avg.sel(depth=slice(0,200))
    varf_bot = varf_time_avg.sel(depth=slice(200,999999))


    mean_top_e = []
    mean_top_d = []
    mean_bot_e = []
    mean_bot_d = []

    var_top_e = []
    var_top_d = []
    var_bot_e = []
    var_bot_d = []

    varf_top_e = []
    varf_top_d = []
    varf_bot_e = []
    varf_bot_d = []

    
    for lat, lon in zip(lats, lons):
        m_lons = int(len(lons)/4) 
        # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
        # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

        if lon < lons[m_lons]:
        # Extrair o perfil em profundidade para o par (lat, lon)
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        elif lon == lons[m_lons]:
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        else:
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

    output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
    os.makedirs(output_dir + f'/curr_comp/{s}/total/mean', exist_ok=True)
    # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
    # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


    # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))


    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.round(np.max(vals_mean),2)

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_u',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_u', cmap='bwr')
    # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
    #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
    
    # vals_var_max = np.round(np.max(vals_var),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/total/var',
    #                 np.arange(0,vals_var_max, vals_var_max/10))

    # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
    #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    # vals_varf_max = np.round(np.max(vals_varf),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/total/var_filt',
    #                 np.arange(0,vals_varf_max, vals_varf_max/10))

    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_u', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_u')

    # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
    # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
    # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
    # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

    # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
    #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
    
    # vals_dif_max = np.round(np.nanmax(vals_dif),2)

    # if np.isnan(np.nan):
    #     vals_dif_max = .05
        
    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/total/var_diff',
    #                 np.arange(0,vals_dif_max,vals_dif_max/10))


####564654984651784653198432189415

s = 0
for section in sections:
    s+=1
    print(f'Iniciando seção {s}')
    mean_time = []
    var_time = []
    varf_time = []
    for year in years:
        reanal = {}
        print('comecando ' + str(year))

        reanal[year] = set_reanalisys_curr_dims(xr.open_mfdataset(model_path + model + '/UV/' + str(year)  + '/*.nc')
                                            , model)

        reanalisys = xr.concat(list(reanal.values()), dim="time")
        # year = 'total'
        # define a linha que eu vou plotar

        delta_lon = section['lon'].values[-1] - section['lon'].values[0]
        delta_lat = section['lat'].values[-1] - section['lat'].values[0]
        theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
        theta_deg = np.degrees(theta_rad)  # Convertendo para graus


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

        # rotacionar direto o reanal_subset pode ser mais facil



        # Aplicar a rotação a cada ponto de grade
        reanal_subset = reanal_subset.chunk(dict(time=-1))

        # along_shore, cross_shore = xr.apply_ufunc(
        #     rotate_current,
        #     reanal_subset['u'],  # Entrada U
        #     reanal_subset['v'],  # Entrada V
        #     theta_deg,           # Ângulo como escalar
        #     input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
        #     output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
        #     vectorize=True,  # Permite a aplicação para todas as grades
        #     dask="parallelized",  # Habilita o processamento paralelo
        #     output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
        # )

        # # Adicionar as componentes rotacionadas ao Dataset
        # along_shore = along_shore.chunk({'depth': 17, 'latitude': 28, 'longitude': 45, 'time': 30})
        # print('comecou a computar os dados along_shore')
        # along_shore = along_shore.compute()
        # print('terminou')
        # reanal_subset['along_shore'] =  cross_shore.transpose("time", "depth", "latitude", "longitude")
        # reanal_subset['cross_shore'] = cross_shore

        # # fazendo a mesma coisa para filtrado:

        reanal_subset['along_shore_filt'] = model_filt.filtra_reanalise_v(reanal_subset)


        # pegar a variância e a média do along_shore
        mean_along = reanal_subset['v'].mean(dim='time')
        var_along = reanal_subset['v'].var(dim='time')

        # pegar a variância do along_shore_filt
        var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

        lats = section['lat'].values
        lons = section['lon'].values

        mean_time.append(mean_along)
        var_time.append(var_along)
        varf_time.append(var_alongf)




        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,200))
        mean_bot= mean_along.sel(depth=slice(200,999999))

        var_top= var_along.sel(depth=slice(0,200))
        var_bot= var_along.sel(depth=slice(200,999999))

        varf_top = var_alongf.sel(depth=slice(0,200))
        varf_bot = var_alongf.sel(depth=slice(200,999999))


        mean_top_e = []
        mean_top_d = []
        mean_bot_e = []
        mean_bot_d = []

        var_top_e = []
        var_top_d = []
        var_bot_e = []
        var_bot_d = []

        varf_top_e = []
        varf_top_d = []
        varf_bot_e = []
        varf_bot_d = []

        
        for lat, lon in zip(lats, lons):
            m_lons = int(len(lons)/4) 
            # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[m_lons]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            elif lon == lons[m_lons]:
                mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            else:
                mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
                mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
                var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
                varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
        os.makedirs(output_dir + f'/curr_comp/{s}/{year}/mean', exist_ok=True)
        # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
        # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


        # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


        # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
        # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

        print('comecando plot ')

        vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                            abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
        
        vals_mean_max = np.round(np.max(vals_mean),2)

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_v',
                            np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/mean_sem_lim_v', cmap='bwr')

        # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
        #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
        
        # vals_var_max = np.round(np.max(vals_var),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var',
        #                 np.arange(0,vals_var_max, vals_var_max/10))

        # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
        #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        # vals_varf_max = np.round(np.max(vals_varf),2)

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_filt',
        #                 np.arange(0,vals_varf_max, vals_varf_max/10))

        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_v', np.arange(0,101,10))
        
        four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                        lons[:m_lons+1], lons[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_perc_sem_lim_v')

        # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
        # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
        # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
        # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

        # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
        #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
        
        # vals_dif_max = np.round(np.nanmax(vals_dif),2)
        
        # if np.isnan(np.nan):
        #     vals_dif_max = .05

        # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
        #                 lons[:m_lons+1], lons[m_lons:],
        #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/{str(year)}/var_diff',
        #                 np.arange(0,vals_dif_max,vals_dif_max/10))
        
    # fazendo pra media do tempo:
    print('total')

    mean_time_concat = xr.concat(mean_time, dim="year")
    var_time_concat = xr.concat(var_time, dim="year")
    varf_time_concat = xr.concat(varf_time, dim="year")

    # Calcular a média ao longo dos anos
    mean_time_avg = mean_time_concat.mean(dim="year")
    var_time_avg = var_time_concat.mean(dim="year")
    varf_time_avg = varf_time_concat.mean(dim="year")

    # Selecionar a variável
    mean_top= mean_time_avg.sel(depth=slice(0,200))
    mean_bot= mean_time_avg.sel(depth=slice(200,999999))

    var_top= var_time_avg.sel(depth=slice(0,200))
    var_bot= var_time_avg.sel(depth=slice(200,999999))

    varf_top = varf_time_avg.sel(depth=slice(0,200))
    varf_bot = varf_time_avg.sel(depth=slice(200,999999))


    mean_top_e = []
    mean_top_d = []
    mean_bot_e = []
    mean_bot_d = []

    var_top_e = []
    var_top_d = []
    var_bot_e = []
    var_bot_d = []

    varf_top_e = []
    varf_top_d = []
    varf_bot_e = []
    varf_bot_d = []

    
    for lat, lon in zip(lats, lons):
        m_lons = int(len(lons)/4) 
        # filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
        # raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

        if lon < lons[m_lons]:
        # Extrair o perfil em profundidade para o par (lat, lon)
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        elif lon == lons[m_lons]:
            mean_top_e.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_e.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_e.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_e.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_e.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_e.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))
    
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        else:
            mean_top_d.append(mean_top.sel(latitude=lat, longitude=lon, method='nearest'))
            mean_bot_d.append(mean_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            var_top_d.append(var_top.sel(latitude=lat, longitude=lon, method='nearest'))
            var_bot_d.append(var_bot.sel(latitude=lat, longitude=lon, method='nearest'))

            varf_top_d.append(varf_top.sel(latitude=lat, longitude=lon, method='nearest'))
            varf_bot_d.append(varf_bot.sel(latitude=lat, longitude=lon, method='nearest'))

    output_dir = f'/home/bcabral/mestrado/fig/secmean/secao{s}/'
    os.makedirs(output_dir + f'/curr_comp/{s}/total/mean', exist_ok=True)
    # os.makedirs(output_dir + f'/var/{str(year)}', exist_ok=True)
    # os.makedirs(output_dir + f'/varf/{str(year)}', exist_ok=True)


    # def four_window_custom(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path, ticks = None):


    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))


    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.round(np.max(vals_mean),2)

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_v',
                        np.arange(-vals_mean_max, vals_mean_max,round((vals_mean_max*2)/10,2)), cmap='bwr')


    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir+f'/curr_comp/{s}/total/mean_sem_lim_v', cmap='bwr')
    # vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
    #                     abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])
    
    # vals_var_max = np.round(np.max(vals_var),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir+f'/curr_comp/{s}/total/var',
    #                 np.arange(0,vals_var_max, vals_var_max/10))

    # vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
    #                     abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    # vals_varf_max = np.round(np.max(vals_varf),2)

    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir+f'/curr_comp/{s}/total/var_filt',
    #                 np.arange(0,vals_varf_max, vals_varf_max/10))

    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_v', np.arange(0,101,10))
    
    four_window_custom_sem_lim(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
                    lons[:m_lons+1], lons[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir+f'/curr_comp/{s}/total/var_perc_v')

    # dif_top_e = np.asarray(var_top_e) - np.asarray(varf_top_e)
    # dif_top_d = np.asarray(var_top_d) - np.asarray(varf_top_d)
    # dif_bot_e = np.asarray(var_bot_e) - np.asarray(varf_bot_e)
    # dif_bot_d = np.asarray(var_bot_d) - np.asarray(varf_bot_d)

    # vals_dif = np.array([abs(np.nanmax(dif_top_e)), abs(np.nanmax(dif_bot_e)), abs(np.nanmax(dif_top_d)), abs(np.nanmax(dif_bot_d)),
    #                     abs(np.nanmin(dif_top_e)), abs(np.nanmin(dif_bot_e)), abs(np.nanmin(dif_top_d)), abs(np.nanmin(dif_bot_d))])
    
    # vals_dif_max = np.round(np.nanmax(vals_dif),2)

    # if np.isnan(np.nan):
    #     vals_dif_max = .05
        
    # four_window_custom(reanal_subset['depth'].sel(depth=slice(0,200)), reanal_subset['depth'].sel(depth=slice(200,10000)),
    #                 lons[:m_lons+1], lons[m_lons:],
    #                 dif_top_e, dif_top_d, dif_bot_e, dif_bot_d, output_dir+f'/curr_comp/{s}/total/var_diff',
    #                 np.arange(0,vals_dif_max,vals_dif_max/10))




 ##################################################
  ##################################################
   ##################################################
    ##################################################
     ##################################################
      ##################################################
       ##################################################
        ##################################################
         ##################################################
          ##################################################
           ##################################################
            ##################################################


print('kept you waiting, huh?')








    ##################################################
    ### Fazendo os plots singulares ##################
    ##################################################
    ## iterar na mao: ################################
    ##################################################

    depths = reanal_subset['depth'].values


    # Pré-carregar todas as coordenadas de latitude e longitude relevantes
    latitudes = section['lat'].values
    longitudes = section['lon'].values

    # Pré-selecionar os dados ao longo das coordenadas necessárias
    reanal_subset_loaded = reanal_subset[['along_shore', 'along_shore_filt']].sel(
        latitude=latitudes, longitude=longitudes, method='nearest'
    ).load()

    # Inicializar os arrays para resultados
    section_along = []
    section_along_filt = []


    ##
    # Coordenadas específicas da seção
    lons = section['lon'].values  # 10 longitudes
    lats = section['lat'].values  # 10 latitudes

    contributions = []
    for t, time_step in enumerate(reanal_subset['time']):
        print(f'Fazendo {t}: {time_step}')

        time_index = t # Índice do tempo desejado (mude conforme necessário)
        time_selected = reanal_subset['time'].isel(time=time_index)

        # Selecionar a variável
        filt_top= reanal_subset['along_shore_filt'].sel(time=time_selected, depth=slice(0,400))
        filt_bot= reanal_subset['along_shore_filt'].sel(time=time_selected, depth=slice(400,999999))

        raw_top= reanal_subset['along_shore'].sel(time=time_selected, depth=slice(0,400))
        raw_bot= reanal_subset['along_shore'].sel(time=time_selected, depth=slice(400,999999))


        filt_top_e = []
        filt_top_d = []
        filt_bot_e = []
        filt_bot_d = []

        raw_top_e = []
        raw_top_d = []
        raw_bot_e = []
        raw_bot_d = []

        
        filt = []
        raw = []

        # pensando no plot com 4 quadros, fazer o subset aqui, antes de entrar no loop
        # tambem vai ser preciso tratar excecoes no loop quando lat ou lon estiver fora do subset
        for lat, lon in zip(lats, lons):
            filt.append(reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon, time=time_selected, method='nearest'))
            raw.append(reanal_subset['along_shore'].sel(latitude=lat, longitude=lon,time=time_selected, method='nearest'))

            if lon < lons[5]:
            # Extrair o perfil em profundidade para o par (lat, lon)
                filt_top_e.append(filt_top.sel(latitude=lat, longitude=lon, method='nearest'))
                filt_bot_e.append(filt_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                raw_top_e.append(raw_top.sel(latitude=lat, longitude=lon, method='nearest'))
                raw_bot_e.append(raw_bot.sel(latitude=lat, longitude=lon, method='nearest'))
            elif lon == lons[5]:
                filt_top_e.append(filt_top.sel(latitude=lat, longitude=lon, method='nearest'))
                filt_bot_e.append(filt_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                raw_top_e.append(raw_top.sel(latitude=lat, longitude=lon, method='nearest'))
                raw_bot_e.append(raw_bot.sel(latitude=lat, longitude=lon, method='nearest'))
        
                filt_top_d.append(filt_top.sel(latitude=lat, longitude=lon, method='nearest'))
                filt_bot_d.append(filt_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                raw_top_d.append(raw_top.sel(latitude=lat, longitude=lon, method='nearest'))
                raw_bot_d.append(raw_bot.sel(latitude=lat, longitude=lon, method='nearest'))
            else:
                filt_top_d.append(filt_top.sel(latitude=lat, longitude=lon, method='nearest'))
                filt_bot_d.append(filt_bot.sel(latitude=lat, longitude=lon, method='nearest'))

                raw_top_d.append(raw_top.sel(latitude=lat, longitude=lon, method='nearest'))
                raw_bot_d.append(raw_bot.sel(latitude=lat, longitude=lon, method='nearest'))

        output_dir = f'/home/bcabral/mestrado/fig/secao{s}/'
        os.makedirs(output_dir + f'/raw/{str(time_step.values)[:4]}', exist_ok=True)
        os.makedirs(output_dir + f'/filt/{str(time_step.values)[:4]}', exist_ok=True)

        four_window_fig(raw_top_e, raw_top_d, raw_bot_e, raw_bot_d, output_dir+f'/raw/{str(time_step.values)[:4]}/{str(time_step.values)[:10]}')

        four_window_fig_filt(filt_top_e, filt_top_d, filt_bot_e, filt_bot_d, output_dir+f'/filt/{str(time_step.values)[:4]}/{str(time_step.values)[:10]}' )


        raw = np.array(raw)
        filt = np.array(filt)
        brute = raw - filt
        # brute = np.where(filt>0, raw-filt, raw + filt)

        great = np.where(np.abs(brute)>np.abs(raw), brute, raw)

        contribution = (abs(filt)/abs(great)) * 100

        contributions.append(contribution)

    contributions= np.array(contributions)

    mean_contribution = np.mean(contributions, axis=0)

    plot_section(lons, depths, mean_contribution, 'Contribuição média das OCCs à corrente', output_dir + f'/contribuicao_{str(time_step.values)[:4]}')

########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################
############################### OLD GARBAGE
# Inicializar arrays para os resultados da seção
section_along = []
section_along_filt = []
for t, time_step in enumerate(reanal_subset['time']):
    print(t)
    time_index = t  # Índice do tempo desejado (mude conforme necessário)
    time_selected = reanal_subset['time'].isel(time=time_index)

    # Selecionar a variável
    variable = 'along_shore_filt'  # Pode ser along_shore, cross_shore, etc.
    data_at_time = reanal_subset[variable].sel(time=time_selected)


    section_along_filt_t = []
    # pensando no plot com 4 quadros, fazer o subset aqui, antes de entrar no loop
    # tambem vai ser preciso tratar excecoes no loop quando lat ou lon estiver fora do subset
    for lat, lon in zip(lats, lons):
        # Extrair o perfil em profundidade para o par (lat, lon)
        profile = data_at_time.sel(latitude=lat, longitude=lon, method='nearest')
        section_along_filt_t.append(profile)

# # Loop pelos tempos
# for t, time_step in enumerate(reanal_subset['time']):
#     print(f"Processando tempo {t}: {str(time_step.values)[:10]}")

#     # Inicializar as matrizes para o tempo atual
#     section_along_t = []
#     section_along_filt_t = []

#     # Iterar pelos pontos da seção (10 pontos)
#     for lon, lat in zip(lons, lats):
#         # Selecionar o ponto específico com todas as profundidades
#         along_point = reanal_subset['along_shore'].isel(time=t).sel(
#             longitude=lon, latitude=lat, method='nearest'
#         )
#         along_filt_point = reanal_subset['along_shore_filt'].isel(time=t).sel(
#             longitude=lon, latitude=lat, method='nearest'
#         )

#         # Adicionar os valores do ponto à matriz da seção
#         section_along_t.append(along_point.values)  # Shape (51,)
#         section_along_filt_t.append(along_filt_point.values)  # Shape (51,)

#     # Converter as listas para numpy arrays (10x51) e salvar no array principal
#     # section_along.append(np.array(section_along_t))  # Shape (10, 51)
#     # section_along_filt.append(np.array(section_along_filt_t))  # Shape (10, 51)

#     # Caminhos para saída
    raw_path = f"{output_dir}/raw/00TESTEsection_{t:03d}.png"
    filt_path = f"{output_dir}/filt/00TESTEsection_{t:03d}.png"

    # Plotar gráficos
    # plot_section(lons, depths, np.array(section_along_t), f"Seção - Tempo: {str(time_step.values)[:10]}", raw_path)
    plot_section(lons, depths, np.array(section_along_filt_t), f"Seção Filtrada - Tempo: {str(time_step.values)[:10]}", filt_path)



# Loop pelos tempos
for t, time_step in enumerate(reanal_subset['time']):
    print(f"Processando tempo {t}: {str(time_step.values)[:10]}")

    # Extrair seções para o tempo atual
    section_along = reanal_subset_loaded['along_shore'].isel(time=t).values
    section_along_filt = reanal_subset_loaded['along_shore_filt'].isel(time=t).values

    # possivelmente vai ter que dar .T quando precisar plotar, pra transpor a matriz

    # Caminhos para saída
    raw_path = f"{output_dir}/raw/section_{t:03d}.png"
    filt_path = f"{output_dir}/filt/section_{t:03d}.png"

    # Plotar gráficos
    plot_section(longitudes, depths, section_along, f"Seção - Tempo: {str(time_step.values)[:10]}", raw_path)
    plot_section(longitudes, depths, section_along_filt, f"Seção Filtrada - Tempo: {str(time_step.values)[:10]}", filt_path)



### bre
for t, time_step in enumerate(reanal_subset['time']):
    reanal_subset['along_shore'].sel(time=time_step).compute()
    print(t)


    for index, row in section.iterrows():
        lon = row['lon']
        lat = row['lat']
        
        # Seleciona os dados em função da longitude e latitude
        section_along.append(reanal_subset['along_shore'].isel(time=t).sel(longitude=lon, latitude=lat, method='nearest').values)
        section_along_filt.append(reanal_subset['along_shore_filt'].isel(time=t).sel(longitude=lon, latitude=lat, method='nearest').values)

        # section_cross.append(reanal_subset['cross_shore'].sel(longitude=lon, latitude=lat, method='nearest').values)


    # Converter para arrays numpy
    section_along = np.array(section_along)
    # section_cross = np.array(section_cross)

    output_dir = '/home/bcabral/mestrado/fig/curr_gif/'
    os.makedirs(output_dir + '/raw', exist_ok=True)
    os.makedirs(output_dir + '/filt', exist_ok=True)


    plt.figure(figsize=(10, 6))

    # Plotar a seção
    plt.contourf(section['lon'], depths, section_along.T, levels=50, cmap='bwr')  # Transpondo para profundidade vs longitude
    plt.colorbar(label='Int. (m/s)')
    plt.title(f"Seção - Tempo: {str(time_step.values)[:10]}")
    plt.xlabel('Longitude')
    plt.ylabel('Profundidade (m)')
    plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
    plt.tight_layout()
    # Salve a figura
    plt.savefig(f"{output_dir}/raw/section_{t:03d}.png")
    plt.close()


    plt.figure(figsize=(10, 6))

    # Plotar a seção
    plt.contourf(section['lon'], depths, section_along_filt.T, levels=50, cmap='bwr')  # Transpondo para profundidade vs longitude
    plt.colorbar(label='Int. (m/s)')
    plt.title(f"Seção - Tempo: {str(time_step.values)[:10]}")
    plt.xlabel('Longitude')
    plt.ylabel('Profundidade (m)')
    plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
    plt.tight_layout()
    # Salve a figura
    plt.savefig(f"{output_dir}/filt/section_{t:03d}.png")
    plt.close()


# # Obter os valores de profundidade

##


output_dir = '/home/bcabral/mestrado/fig/curr_gif/'
os.makedirs(output_dir, exist_ok=True)

# Plotar cada passo de tempo

for t, time_step in enumerate(reanal_subset['time']):
    print(t)
    # Extraia a seção no tempo atual
    section = reanal_subset['along_shore'].isel(time=t).mean(dim='latitude')
    
    # Crie a figura
    plt.figure(figsize=(10, 6))
    
    # Crie o plot de contorno
    lon, depth = np.meshgrid(reanal_subset['longitude'], reanal_subset['depth'])
    plt.contourf(lon, depth, section.T, levels=20, cmap="bwr", extend='both')
    
    # Ajuste dos eixos e rótulos
    plt.gca().invert_yaxis()  # Inverta o eixo Y para profundidade
    plt.title(f"Seção - Tempo: {str(time_step.values)[:10]}")
    plt.xlabel("Longitude")
    plt.ylabel("Profundidade (m)")
    plt.colorbar(label="Velocidade Along-Shore (m/s)")
    
    # Salve a figura
    plt.savefig(f"{output_dir}/section_{t:03d}.png")
    plt.close()

# GIF!
from PIL import Image
import glob

# Criar o GIF
frames = []
for file in sorted(glob.glob(f"{output_dir}/section_*.png")):
    frames.append(Image.open(file))
frames[0].save("sections.gif", save_all=True, append_images=frames[1:], duration=200, loop=0)

#PAGINA HTML

with open("sections.html", "w") as f:
    f.write('<html><body>\n')
    for file in sorted(glob.glob(f"{output_dir}/section_*.png")):
        f.write(f'<img src="{file}" style="max-width: 100%; display: block;">\n')
    f.write('</body></html>')


####

# # VELHO
# for t, time_step in enumerate(reanal_subset['time']):
#     section = reanal_subset['along_shore'].isel(time=t).mean(dim='latitude')  # Média na profundidade
#     plt.figure(figsize=(8, 4))
#     section.plot(cmap="coolwarm", robust=True)
#     plt.title(f"Seção - Tempo: {str(time_step.values)[:10]}")
#     plt.xlabel("Longitude")
#     plt.ylabel("Latitude")
#     plt.savefig(f"{output_dir}/section_{t:03d}.png")
#     plt.close()


#     plt.figure(figsize=(10, 6))

#     # Plotar a seção
#     plt.contourf(section['lon'], depths, along_shore.T, levels=50, cmap='bwr')  # Transpondo para profundidade vs longitude
#     plt.colorbar(label='Int. (m/s)')
#     plt.title('Média de velocidade normal à seção')
#     plt.xlabel('Longitude')
#     plt.ylabel('Profundidade (m)')
#     plt.gca().invert_yaxis()  # Inverter o eixo y para profundidade
#     plt.tight_layout()
#     plt.savefig(f'/home/bcabral/mestrado/fig/curr_section_raw/{model}_{num_sec}')
#     plt.close()


# ####


# ##########

# # Rechunk a dimensão 'time' para evitar conflitos
# reanal_subset = reanal_subset.chunk(dict(time=-1))

# # Aplicar a rotação com apply_ufunc
# u_prime, v_prime = xr.apply_ufunc(
#     rotate_current_series,
#     reanal_subset['u'],  # Entrada U
#     reanal_subset['v'],  # Entrada V
#     theta_deg,           # Ângulo como escalar
#     input_core_dims=[["time"], ["time"], []],  # Dimensão relevante é apenas o tempo
#     output_core_dims=[["time"], ["time"]],    # Saídas têm apenas dimensão de tempo
#     vectorize=True,  # Permite a aplicação para todas as grades
#     dask="parallelized",  # Habilita o processamento paralelo
#     output_dtypes=[reanal_subset['u'].dtype, reanal_subset['v'].dtype]
# )

# # Adicionar as componentes rotacionadas ao Dataset
# reanal_subset['u_prime'] = u_prime
# reanal_subset['v_prime'] = v_prime

# ###########

# plota a secao media de cada ano
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
