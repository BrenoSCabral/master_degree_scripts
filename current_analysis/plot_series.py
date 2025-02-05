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


def curr_window(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path,ticks, cmap='cool_r'):

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
    cbar = fig.colorbar(im1, cax=cbar_ax, label='', extend='both')

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    fig.text(0.05, 0.5, 'Profundidade (m)', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.text(0.5, 0.05, 'Distância da costa (km)', ha='center', va='center', rotation='horizontal', fontsize=12)

    plt.savefig(path)






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
        if year != 2014:
            continue
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


        # Supondo que 'reanal_subset' seja um xarray.Dataset e 'section' seja um DataFrame ou dict com as latitudes.
        i_start = 0
        lat_start = section['lat'][i_start]  # Latitude no começo
        lon_start = section['lon'][i_start]  # Latitude no começo

        na_int = (reanal_subset['along_shore_filt'].sel(latitude=lat_start, 
                                               longitude=lon_start, depth=0, method='nearest').mean() < 10).values
        while not na_int:
            i_start +=1
            lat_start = section['lat'][i_start]  # Latitude no começo
            lon_start = section['lon'][i_start]  # Latitude no começo

            na_int = (reanal_subset['along_shore_filt'].sel(latitude=lat_start, 
                                                longitude=lon_start, depth=0, method='nearest').mean() < 10).values
        

        lat_mid = section['lat'][len(section['lat']) // 2]  # Latitude no meio
        lat_end = section['lat'][len(section['lat'])-1]  # Latitude no final

        lon_mid = section['lon'][len(section['lon']) // 2]  # Latitude no meio
        lon_end = section['lon'][len(section['lon'])-1]  # Latitude no final



        # Criando a figura com 1 coluna e 3 linhas de subplots
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8), sharex=True)

        # Plotando os dados para cada latitude em um subplot
        axes[0].plot(reanal_subset['along_shore_filt'].sel(latitude=lat_start, longitude=lon_start, depth=0, method='nearest'), label=f'Latitude {lat_start}')
        axes[0].set_title(f'2014-superficie\nPonto Costeiro')
        axes[0].set_ylabel('Along Shore Filtered')
        axes[0].grid(True)

        axes[1].plot(reanal_subset['along_shore_filt'].sel(latitude=lat_mid, longitude=lon_mid, depth=0, method='nearest'), label=f'Latitude {lat_mid}')
        axes[1].set_title(f'Ponto médio')
        axes[1].set_ylabel('Along Shore Filtered')
        axes[1].grid(True)

        axes[2].plot(reanal_subset['along_shore_filt'].sel(latitude=lat_end, longitude=lon_end, depth=0, method='nearest'), label=f'Latitude {lat_end}')
        axes[2].set_title(f'Ponto Final')
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Along Shore Filtered')
        axes[2].grid(True)


        for i, (lat, label) in enumerate([(lat_start, 'Ponto inicial'), (lat_mid, 'Ponto médio'), (lat_end, 'Ponto Final')]):
            serie = reanal_subset['along_shore_filt'].sel(latitude=lat, longitude=lon_start, depth=0, method='nearest')

            # Cálculo do desvio padrão e coeficiente de variação
            desvio_padrao = serie.std().values
            variabilidade = serie.var().values

            # Adicionando o desvio padrão e a variabilidade no canto superior direito
            axes[i].text(0.95, 0.95, f'Desvio Padrão: {desvio_padrao:.2f}\nCoef. de Var.: {variabilidade:.2f}', 
                        transform=axes[i].transAxes, ha='right', va='top', fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))


        # Ajustando o layout para evitar sobreposição
        plt.tight_layout()
        plt.savefig(f'/home/bcabral/mestrado/fig/curr/{s}/along_series.png')
        # plt.show()
