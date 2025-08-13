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


def curr_window(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d, path,ticks, l_contour, cmap='cool_r', cbar_label='', cbar_ticks=[]):
    fig, ax = plt.subplots(2,2, figsize=(12,9))


    # dep_sup = reanal_subset['depth'].sel(depth=slice(0,400))
    # dep_bot = reanal_subset['depth'].sel(depth=slice(400,10000))

    # lon_e = lons[:6] # reanal_subset['longitude'].sel(longitude=slice(lons[0], lons[5]))
    # lon_d = lons[5:] # reanal_subset['longitude'].sel(longitude=slice(lons[5], lons[9]))

    im1 = ax[0,0].contourf(lon_e, -dep_sup, np.array(top_e).T, cmap=cmap, levels=ticks)
    

    ax[0,0].set_facecolor([0,0,0,0.6]) 

    if cmap == 'bwr':
        contour_color = 'black'
    else:
        contour_color = 'lightgrey'


    cs1 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    ax[0,0].clabel(cs1, inline=True,fontsize=10)

    # ax[0,0].set_xlim(-55, -45)
    #ax[0,0].set_xlabel("Longitude (°)")
    # ax[0,0].set_ylabel("Depth (m)")
    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    # im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1],
    # shading='gouraud')
    im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, levels=ticks)
    ax[0,1].set_facecolor([0,0,0,0.6]) 

    cs1 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    ax[0,1].clabel(cs1, inline=True,fontsize=10)

    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, np.array(bot_e).T, cmap=cmap, levels=ticks)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    ax[1,0].clabel(cs2, inline=True,fontsize=10)


    # ax[1,0].set_ylabel("Depth (m)")




    ## Direita Inferior -----------------------------------



    # im2 = ax[1,1].pcolormesh(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1], shading='gouraud')
    im2 = ax[1,1].contourf(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, levels=ticks)
    ax[1,1].set_facecolor([0,0,0,0.6]) 

    cs2 = ax[1,1].contour(lon_d, - dep_bot, np.array(bot_d).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    ax[1,1].clabel(cs2, inline=True,fontsize=10)

    ax[1,1].set_yticks([])

    # Adicionar colorbar à direita
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both', ticks=cbar_ticks)
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    fig.text(0.05, 0.5, 'Depth (m)', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.text(0.5, 0.05, 'Distance from the Coast (km)', ha='center', va='center', rotation='horizontal', fontsize=12)

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
    if s == 1:
        section = section[2:].reset_index().drop('index', axis=1)
    if s ==2:
        section = section[4:].reset_index().drop('index', axis=1)
    if s == 3:
        section = section[3:].reset_index().drop('index', axis=1) # corta uma parte de terra especificamente pra esse ponto
    elif s == 4:
        section = section[3:].reset_index().drop('index', axis=1)
    elif s ==5:
        section = section[2:].reset_index().drop('index', axis=1)
    elif s == 6:
        section = section[4:].reset_index().drop('index', axis=1)
    elif s ==7:
        section = section[9:].reset_index().drop('index', axis=1)
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



        div_d = 200

        if s == 2:
            div_d =175
        elif s==1 or s==3:
            div_d = 160
        elif s==4:
            div_d = 40
        elif s==5:
            div_d = 75
        elif s==6 or s==7:
            div_d = 50
            # div_d = 17.5

        # Selecionar a variável
        mean_top= mean_along.sel(depth=slice(0,div_d))
        mean_bot= mean_along.sel(depth=slice(div_d,1000))

        var_top= var_along.sel(depth=slice(0,div_d))
        var_bot= var_along.sel(depth=slice(div_d,1000))

        varf_top = var_alongf.sel(depth=slice(0,div_d))
        varf_bot = var_alongf.sel(depth=slice(div_d,1000))


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
            if s ==2:
                m_lons = 82
            if s == 3:
                m_lons = 32
            elif s ==4:
                m_lons = 23
            elif s ==5:
                m_lons = 9
            elif s ==6:
                m_lons =6
            elif s ==7:
                m_lons = 8
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

        output_dir = f'/home/bcabral/mestrado/fig/curr_0410'
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

        l_contour = np.arange(-np.round(vals_mean_max, 1), vals_mean_max, .1)
        

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                            mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir + f'/{s}/{year}/mean',
                            np.arange(-vals_mean_max, vals_mean_max,(vals_mean_max)/1000), l_contour=l_contour, cmap='bwr', cbar_label='(m/s)')

        vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
                            abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])

        vals_var_max = np.max(vals_var)


        if s ==1 or s==2 or s==7:
            l_contour = np.arange(0, 0.1, .01)
        elif s==3 or s==4:
            l_contour = np.arange(0, 0.05, .005)
        elif s==5:
            l_contour = np.arange(0, 0.02, .002)
        elif s==6:
            l_contour = np.arange(0, 0.07, .005)



        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir + f'/{s}/{year}/var',
                        np.arange(0,vals_var_max, vals_var_max/1000), l_contour=l_contour, cbar_label='(m/s)\u00B2', cmap='inferno')
        

        l_contour = np.arange(0, 101, 20)

        perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
        perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
        perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
        perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir + f'/{s}/{year}/perc', np.arange(0,100,.1),l_contour=l_contour, cmap='inferno', cbar_label='%')



        if s ==1 or s==2 or s==7:
            l_contour = np.arange(0, 1, .1)
        elif s==3 or s==4:
            l_contour = np.arange(0, 1, .1)
        elif s==5:
            l_contour = np.arange(0, 1, .1)
        elif s==6:
            l_contour = np.arange(0, 1, .1)


        vals_std_max = np.sqrt(vals_var_max)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        np.sqrt(var_top_e), np.sqrt(var_top_d), np.sqrt(var_bot_e), np.sqrt(var_bot_d),
                        output_dir + f'/{s}/{year}/std',
                        np.arange(0,vals_std_max, vals_std_max/1000), cmap='inferno', l_contour=l_contour, cbar_label='(m/s)')


        if s ==1 or s==2 or s==7:
            l_contour = np.arange(0, 0.1, .01)
        elif s==3 or s==4:
            l_contour = np.arange(0, 0.05, .005)
        elif s==5:
            l_contour = np.arange(0, 0.02, .002)
        elif s==6:
            l_contour = np.arange(0, 0.07, .005)


        vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
                            abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
        
        vals_varf_max = np.max(vals_varf)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir + f'/{s}/{year}/varf',
                        np.arange(0,vals_varf_max, vals_varf_max/1000), l_contour=l_contour, cmap='inferno', cbar_label='(m/s)\u00B2')

        vals_stdf_max = np.sqrt(vals_varf_max)

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        np.sqrt(varf_top_e), np.sqrt(varf_top_d),
                        np.sqrt(varf_bot_e), np.sqrt(varf_bot_d),
                        output_dir + f'/{s}/{year}/stdf',
                        np.arange(0,vals_stdf_max, vals_stdf_max/1000), cmap='inferno', l_contour=l_contour, cbar_label='(m/s)') 


        l_contour = np.arange(0, 101, 20)

        perc_top_e_st = (np.asarray(np.sqrt(varf_top_e))/np.asarray(np.sqrt(var_top_e))) * 100
        perc_top_d_st = (np.asarray(np.sqrt(varf_top_d))/np.asarray(np.sqrt(var_top_d))) * 100
        perc_bot_e_st = (np.asarray(np.sqrt(varf_bot_e))/np.asarray(np.sqrt(var_bot_e))) * 100
        perc_bot_d_st = (np.asarray(np.sqrt(varf_bot_d))/np.asarray(np.sqrt(var_bot_d))) * 100

        curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                        dist[:m_lons+1], dist[m_lons:],
                        perc_top_e_st, perc_top_d_st, perc_bot_e_st, perc_bot_d_st,
                        output_dir + f'/{s}/{year}/perc_st', np.arange(0,100,.1),
                        cbar_label='%', l_contour=l_contour, cmap = 'inferno')
        


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
    mean_top= mean_time_avg.sel(depth=slice(0,div_d))
    mean_bot= mean_time_avg.sel(depth=slice(div_d,999999))

    var_top= var_time_avg.sel(depth=slice(0,div_d))
    var_bot= var_time_avg.sel(depth=slice(div_d,999999))

    varf_top = varf_time_avg.sel(depth=slice(0,div_d))
    varf_bot = varf_time_avg.sel(depth=slice(div_d,999999))


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
        if s ==2:
            m_lons = 82
        elif s == 3:
            m_lons = 32
        elif s ==4:
            m_lons = 23
        elif s ==5:
            m_lons = 9
        elif s ==6:
            m_lons =6
        elif s ==7:
            m_lons = 8

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

    output_dir = f'/home/bcabral/mestrado/fig/curr_0410'
    os.makedirs(output_dir + f'/{s}/', exist_ok=True)

    ########################################## IMPORTANTE
    # AJUSTAR TODOS OS VALORES DE COLORBAR
    # AJUSTAR OS TICKS DE COLORBAR NA VARIAVEL cbar_ticks NA FUNCAO
    # como so os resultados totais provavelmente serao usados, so precisa nesse ultimo plot "geral"
    ########################################## IMPORTANTE


    vals_mean = np.array([abs(np.nanmax(mean_top)), abs(np.nanmax(mean_bot)),
                        abs(np.nanmin(mean_top)), abs(np.nanmin(mean_bot))])
    
    vals_mean_max = np.max(vals_mean)

    # l_contour = np.arange(-np.round(vals_mean_max, 1), np.round(vals_mean_max, 1), .1)
    contours = {
        1 : np.round(np.arange(-.3, .4, .1), 1),
        2 : np.round(np.arange(-.3, .4, .1), 1),
        3 : np.round(np.arange(-.3, .4, .1), 1),
        4 : np.round(np.arange(-.25, .3, .05), 2),
        5 : np.round(np.arange(-.15, .16, .05), 2),
        6 :  np.round(np.arange(-.5, .6, .1), 1), 
        7 : np.round(np.arange(-.12, .13, .06), 2)
    }

    l_contour = contours[s]
    cbar_ticks = contours[s]


    # depois eh necessario repensar na questao do intervalo

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                        mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, output_dir + f'/{s}/mean',
                        # np.arange(-vals_mean_max, vals_mean_max + (vals_mean_max)/1000,(vals_mean_max)/1000),
                        np.arange(contours[s][0], contours[s][-1] + contours[s][-1]/999, contours[s][-1]/10000), 
                        l_contour=l_contour, cmap='bwr', cbar_label='(m/s)', cbar_ticks=cbar_ticks)




    vals_var = np.array([abs(np.nanmax(var_top)), abs(np.nanmax(var_bot)),
                        abs(np.nanmin(var_top)), abs(np.nanmin(var_bot))])

    vals_var_max = np.max(vals_var)


    if s ==1 or s==2 or s==7:
        l_contour = np.arange(0, 0.1, .01)
    elif s==3 or s==4:
        l_contour = np.arange(0, 0.05, .005)
    elif s==5:
        l_contour = np.arange(0, 0.02, .002)
    elif s==6:
        l_contour = np.arange(0, 0.07, .005)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    var_top_e, var_top_d, var_bot_e, var_bot_d, output_dir + f'/{s}/var',
                    np.arange(0,vals_var_max + vals_var_max/999, vals_var_max/1000), l_contour=l_contour, cmap = 'inferno', cbar_label='(m/s)\u00B2')
    

    contours = {
        1 : np.round(np.arange(0, .41, .1), 1),
        2 : np.round(np.arange(0, .26, .05), 2),
        3 : np.round(np.arange(0, .26, .05), 2),
        4 : np.round(np.arange(0, .26, .05), 2),
        5 : np.round(np.arange(0, .13, .03), 2),
        6 :  np.round(np.arange(0, .31, .1), 1), 
        7 : np.round(np.arange(0, .36, .05), 2)
    }

    l_contour = contours[s]
    cbar_ticks = contours[s]


    vals_std_max = np.sqrt(vals_var_max)


    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    np.sqrt(var_top_e), np.sqrt(var_top_d),
                    np.sqrt(var_bot_e), np.sqrt(var_bot_d),
                    output_dir + f'/{s}/std',
                    np.arange(0,vals_std_max + vals_std_max/999, vals_std_max/1000), l_contour=l_contour,
                     cbar_ticks=cbar_ticks, cmap='inferno', cbar_label='(m/s)')

    vals_varf = np.array([abs(np.nanmax(varf_top)), abs(np.nanmax(varf_bot)),
                        abs(np.nanmin(varf_top)), abs(np.nanmin(varf_bot))])
    
    vals_varf_max = np.max(vals_varf)

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, output_dir + f'/{s}/varf',
                    np.arange(0,vals_varf_max + vals_varf_max/999, vals_varf_max/1000), l_contour=l_contour, cmap='inferno',  cbar_label='(m/s)\u00B2')
    
    vals_stdf_max = np.sqrt(vals_varf_max)


    contours = {
        1 : np.round(np.arange(0, .36, .05), 2),
        2 : np.round(np.arange(0, .21, .05), 2),
        3 : np.round(np.arange(0, .16, .03), 2),
        4 : np.round(np.arange(0, .21, .05), 2),
        5 : np.round(np.arange(0, .07, .01), 2),
        6 :  np.round(np.arange(0, .13, .03), 2), 
        7 : np.round(np.arange(0, .25, .04), 2)
    }

    l_contour = contours[s]
    cbar_ticks = contours[s]

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    np.sqrt(varf_top_e), np.sqrt(varf_top_d),
                    np.sqrt(varf_bot_e), np.sqrt(varf_bot_d),
                    output_dir + f'/{s}/stdf',
                    np.arange(0,vals_stdf_max + vals_stdf_max/999, vals_stdf_max/1000), l_contour=l_contour,
                    cbar_ticks=cbar_ticks, cmap='inferno', cbar_label='(m/s)')


    l_contour = np.arange(0, 101, 20)
    cbar_ticks = np.arange(0, 101, 10)


    perc_top_e = (np.asarray(varf_top_e)/np.asarray(var_top_e)) * 100
    perc_top_d = (np.asarray(varf_top_d)/np.asarray(var_top_d)) * 100
    perc_bot_e = (np.asarray(varf_bot_e)/np.asarray(var_bot_e)) * 100
    perc_bot_d = (np.asarray(varf_bot_d)/np.asarray(var_bot_d)) * 100

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    perc_top_e, perc_top_d, perc_bot_e, perc_bot_d, output_dir + f'/{s}/perc', np.arange(0,100,.1),
                    cbar_ticks=cbar_ticks, l_contour=l_contour, cmap='inferno', cbar_label='%')
    



    perc_top_e_st = (np.asarray(np.sqrt(varf_top_e))/np.asarray(np.sqrt(var_top_e))) * 100
    perc_top_d_st = (np.asarray(np.sqrt(varf_top_d))/np.asarray(np.sqrt(var_top_d))) * 100
    perc_bot_e_st = (np.asarray(np.sqrt(varf_bot_e))/np.asarray(np.sqrt(var_bot_e))) * 100
    perc_bot_d_st = (np.asarray(np.sqrt(varf_bot_d))/np.asarray(np.sqrt(var_bot_d))) * 100

    curr_window(reanal_subset['depth'].sel(depth=slice(0,div_d)), reanal_subset['depth'].sel(depth=slice(div_d,10000)),
                    dist[:m_lons+1], dist[m_lons:],
                    perc_top_e_st, perc_top_d_st, perc_bot_e_st, perc_bot_d_st, output_dir + f'/{s}/perc_st', np.arange(0,100,.1), l_contour=l_contour, cmap='inferno', cbar_label='%')




