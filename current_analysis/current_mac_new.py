# arquivo feito pra pegar os resultados de corrente pra reanalise e filtra-los
import os
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import sys
import pandas as pd
import datetime
import math
from geopy.distance import geodesic



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
from curr_sec_map import subplot_map

# import plot_hovmoller as ph
import model_filt

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

model = 'BRAN'


def remove_duplicate_points(dataarray):
    df_coords = pd.DataFrame({
        'lat': dataarray.latitude.values,
        'lon': dataarray.longitude.values
    })
    unique_indices = df_coords.drop_duplicates().index.to_numpy()
    return dataarray.isel(points=unique_indices)



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


def curr_window(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d,
                path,ticks, l_contour, bathy_e, bathy_d, cmap='cool_r', cbar_label='', cbar_ticks=[]):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    im1 = ax[0,0].contourf(lon_e, -dep_sup, top_e, cmap=cmap, levels=ticks)
    

    # ax[1,0] = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree()) 
    
    # ax[1,0] = subplot_map(ax[1,0], s)


    # Obtém a posição do eixo original para sobrepor o GeoAxes
    left, bottom, width, height = ax[1,0].get_position().bounds
    left = left + .03
    bottom = bottom +.04
    height = height -.04
    width = width - .03

    # Cria um novo GeoAxes com a mesma posição e projeção geográfica
    ax_geo = fig.add_axes(
        [left, bottom, width, height],
        projection=ccrs.PlateCarree()  # Projeção desejada
    )


    # Define o fundo do GeoAxes como transparente
    # ax_geo.patch.set_alpha(0)  # Importante para ver o eixo original
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])


    ax_geo = subplot_map(ax_geo, s)

    ax_geo.set_zorder(2)  # Coloca o mapa acima do eixo original
    ax[1,0].set_zorder(1)  # Mantém o eixo original atrás

    ax[0,0].set_facecolor([0,0,0,0.6])

    if cmap == 'bwr':
        contour_color = 'black'
    else:
        contour_color = 'lightgrey'

    if l_contour[int(len(l_contour)/2-.5)] == 0:
        l_contour = np.delete(l_contour, int(len(l_contour)/2-.5))
        ax[0,0].contour(lon_e, -dep_sup, top_e, colors=contour_color ,linewidths=1, levels=np.asarray([0]))
        ax[0,1].contour(lon_d, -dep_sup, top_d, colors=contour_color ,linewidths=1, levels=np.asarray([0]))
        ax[1,1].contour(lon_d, -dep_bot, bot_d, colors=contour_color ,linewidths=1, levels=np.asarray([0]))


    elif len(l_contour) == 2:
        try:
            if np.nanmin(top_e) <20 and np.nanmax(top_e) > 20:
                cs0 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='white',linewidths=1, levels=[20])
                ax[0,0].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro ES %')

        try:
            if np.nanmin(top_d) <20 and np.nanmax(top_d) > 20:
                cs0 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='white',linewidths=1, levels=[20])
                ax[0,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DS %')


        try:
            if np.nanmin(bot_d) <20 and np.nanmax(bot_d) > 20:
                cs0 = ax[1,1].contour(lon_d, -dep_bot, np.array(bot_d).T, colors='white',linewidths=1, levels=[20])
                ax[1,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DI %')



        try:
            if np.nanmin(top_e) <50 and np.nanmax(top_e) > 50:
                cs0 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black',linewidths=1, levels=[50])
                ax[0,0].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro ES %')

        try:
            if np.nanmin(top_d) <50 and np.nanmax(top_d) > 50:
                cs0 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black',linewidths=1, levels=[50])
                ax[0,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DS %')


        try:
            if np.nanmin(bot_d) <50 and np.nanmax(bot_d) > 50:
                cs0 = ax[1,1].contour(lon_d, -dep_bot, np.array(bot_d).T, colors='black',linewidths=1, levels=[50])
                ax[1,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DI %')




        # l_contour = np.array([])


    else:

        cs1 = ax[0,0].contour(lon_e, -dep_sup, top_e, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[0,0].clabel(cs1, inline=True,fontsize=10)
    
        cs1 = ax[0,1].contour(lon_d, -dep_sup, top_d, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[0,1].clabel(cs1, inline=True,fontsize=10)

        cs2 = ax[1,1].contour(lon_d, - dep_bot, bot_d, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[1,1].clabel(cs2, inline=True,fontsize=10)


    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    # im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1],
    # shading='gouraud')
    im1 = ax[0,1].contourf(lon_d, -dep_sup, top_d, cmap=cmap, levels=ticks)
    ax[0,1].set_facecolor([0,0,0,0.6]) 


    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])

    # cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    # ax[1,0].clabel(cs2, inline=True,fontsize=10)


    # ax[1,0].set_ylabel("Depth (m)")

    ax[1,0].set_facecolor([0,0,0,0.6])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, bot_e, cmap=cmap, levels=ticks)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    try:
        cs2 = ax[1,0].contour(lon_e, -dep_bot, bot_e, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[1,0].clabel(cs2, inline=True,fontsize=10)
    except Exception as e:
        print('deu ruim no cs2')


    # ax[1,0].set_ylabel("Depth (m)")




    ## Direita Inferior -----------------------------------



    # im2 = ax[1,1].pcolormesh(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1], shading='gouraud')
    im2 = ax[1,1].contourf(lon_d, - dep_bot, bot_d, cmap=cmap, levels=ticks)
    ax[1,1].set_facecolor([0,0,0,0.6]) 




    ax[1,1].set_yticks([])

    # Adicionar colorbar à direita
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both', ticks=cbar_ticks)
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    fig.text(0.05, 0.5, 'Depth (m)', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.text(0.5, 0.05, 'Distance from the coast (km)', ha='center', va='center', rotation='horizontal', fontsize=12)



    #### Aqui botando a batimetria

    # ax[0][0].plot(lon_e, bathy_e.data,
    #               'r--')
    

    # ax[0][1].plot(lon_d, bathy_d.data,
    #             'r--')
    

    # ax[1][1].plot(lon_d, bathy_d.data,
    #               'r--')

    
    # ax[1][0].plot(lon_e, bathy_e.data,
    #               'r--')
    
    # sup_lim = (-dep_sup[-1], -dep_sup[0])
    # bot_lim = (-dep_bot[-1], -dep_bot[0])




    # ax[0][0].set_ylim(sup_lim)
    
    # ax[1][0].set_ylim(bot_lim)

    # ax[0][1].set_ylim(sup_lim)
    # ax[1][1].set_ylim(bot_lim)

    # plt.savefig('/Users/breno/mestrado/tudo_adj_t1.png')
    plt.savefig(path + '.png')


def extract_variable(ds, lat_da, lon_da, d_top, d_bot, m_lons):

    ds_top = ds.sel(latitude=lat_da, longitude=lon_da, depth=d_top, method="nearest")  # shape (depth, points)
    ds_bot = ds.sel(latitude=lat_da, longitude=lon_da, depth=d_bot, method="nearest")  # shape (depth, points)

    ds_vals = ds.sel(latitude=lat_da, longitude=lon_da, method="nearest")  # shape (depth, points)
    ds_vals['depth'] = - ds_vals['depth']
    ds_vals = remove_duplicate_points(ds_vals)


    # Aplicar:
    ds_top = remove_duplicate_points(ds_top)
    ds_bot = remove_duplicate_points(ds_bot)

    longs = ds_top.longitude

    lon_ref = lons[m_lons]

    # Cria máscaras booleanas
    mask_e = longs <= lon_ref
    mask_d = longs > lon_ref


    # Usa as máscaras para fatiar os dados
    ds_top_e = ds_top[:, mask_e].values
    ds_top_d = ds_top[:, mask_d].values

    ds_bot_e = ds_bot[:, mask_e].values
    ds_bot_d = ds_bot[:, mask_d].values

    div_ref = int(mask_e.sum()) # pega o indice de divisao

    lat_points = ds_top.latitude.values
    lon_points = ds_top.longitude.values

    return ds_top_e, ds_top_d, ds_bot_e, ds_bot_d, div_ref, lat_points, lon_points

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

    reanal_subset =  xr.open_mfdataset(f'/Users/breno/mestrado/files/section{s}.nc')

    delta_lon = section['lon'].values[-1] - section['lon'].values[0]
    delta_lat = section['lat'].values[-1] - section['lat'].values[0]
    theta_rad = np.arctan2(delta_lat, delta_lon) + np.pi/2# Ângulo em radianos
    theta_deg = np.degrees(theta_rad)  # Convertendo para graus


    lat_i = section['lat'].min() - 0.15
    lat_f = section['lat'].max() + 0.15
    lon_i = section['lon'].min() - 0.15
    lon_f = section['lon'].max() + 0.15



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

    mean_alongf = reanal_subset['along_shore_filt'].mean(dim='time')

    # pegar a variância do along_shore_filt
    var_alongf = reanal_subset['along_shore_filt'].var(dim='time')

    lats = section['lat'].values
    lons = section['lon'].values


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

    d_top = mean_along.sel(depth=slice(0,div_d)).depth.values
    d_bot = mean_along.sel(depth=slice(div_d,1000)).depth.values

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


    lons_e = []
    lons_d = []
    lat_used = []

    for lon in lons:
        if lon < lons[m_lons]:
            lons_e.append(lon)
        elif lon == lons[m_lons]:
            lons_e.append(lon)
            lons_d.append(lon)
        else:
            lons_d.append(lon)

        lat_used.append(lat)  

    # Transforma as listas de lon para array
    lons_e = np.array(lons_e)
    lons_d = np.array(lons_d)



    mean_test = mean_along.sel(latitude=lat, longitude=xr.DataArray(lons_e, dims= "lon"), depth=d_top, method="nearest")


    lon_ref = lons[m_lons] # longitude q separa os plots

    # Inicializa listas separadas
    lats_e, lons_e = [], []
    lats_d, lons_d = [], []

    # Divide os pontos em E e D
    for lat, lon in zip(lats, lons):
        if lon < lon_ref:
            lats_e.append(lat)
            lons_e.append(lon)
        elif lon == lon_ref:
            # ponto de separação entra nos dois lados, se quiser
            lats_e.append(lat)
            lons_e.append(lon)
            lats_d.append(lat)
            lons_d.append(lon)
        else:
            lats_d.append(lat)
            lons_d.append(lon)

    # Converte para DataArray
    lat_e = xr.DataArray(lats_e, dims="points")
    lon_e = xr.DataArray(lons_e, dims="points")

    lat_d = xr.DataArray(lats_d, dims="points")
    lon_d = xr.DataArray(lons_d, dims="points")

    # Extrai os perfis verticais para cada lado

    lat_da = xr.DataArray(lats, dims="points")
    lon_da = xr.DataArray(lons, dims="points")


    var_top_e, var_top_d, var_bot_e, var_bot_d, div_ref, lat_points, lon_points = extract_variable(var_along, lat_da, lon_da, d_top, d_bot, m_lons)
    varf_top_e, varf_top_d, varf_bot_e, varf_bot_d, div_ref, lat_points, lon_points = extract_variable(var_alongf, lat_da, lon_da, d_top, d_bot, m_lons)
    meanf_top_e, meanf_top_d, meanf_bot_e, meanf_bot_d, div_ref, lat_points, lon_points = extract_variable(mean_alongf, lat_da, lon_da, d_top, d_bot, m_lons)
    mean_top_e, mean_top_d, mean_bot_e, mean_bot_d, div_ref, lat_points, lon_points = extract_variable(mean_along, lat_da, lon_da, d_top, d_bot, m_lons)



    bathy_file_path = '/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc'
    bathy_ds_raw = xr.open_dataset(bathy_file_path)


    ##### colocar a batimetria

    points_ds = xr.Dataset(
        coords={
            'points': np.arange(len(lat_points)),
            'lat': ('points', lat_points.data),  # ou .values
            'lon': ('points', lon_points.data),
        }
    )


    bathy_interp = bathy_ds_raw['elevation'].interp(
        lon=('points', lon_points.data),
        lat=('points', lat_points.data),
        method='nearest'
    )

    lat_section = lat_points
    lon_section = lon_points

    dist_km = [0]
    for i in range(1, len(lat_section)):
        dist = geodesic((lat_section[i-1], lon_section[i-1]), (lat_section[i], lon_section[i])).km
        dist_km.append(dist_km[-1] + dist)
    dist_km = np.array(dist_km)


    dist_km_e = dist_km[:div_ref]
    dist_km_d = dist_km[div_ref:]

    ## PLOT

    output_dir = '/Users/breno/mestrado/curr_sec_s_bat/'

    # Intensidade media

    contours = {
        1 : np.round(np.arange(-.35, .36, .05), 2),
        2 : np.round(np.arange(-.3, .31, .05), 2),
        3 : np.round(np.arange(-.3, .31, .05), 2),
        4 : np.round(np.arange(-.25, .26, .05), 2),
        5 : np.round(np.arange(-.15, .16, .05), 2),
        6 :  np.round(np.arange(-.5, .51, .05), 2), 
        7 : np.round(np.arange(-1, 1.1, .1), 2)
    }

    l_contour = contours[s]
    cbar_ticks = contours[s]


    ticks = {
        1: np.arange(-.3, .3 + (.3)/1000,(.3)/1000),
        2: np.arange(-.3, .3 + (.3)/1000,(.3)/1000),
        3: np.arange(-.25, .25 + (.25)/1000,(.25)/1000),
        4: np.arange(-.25, .25 + (.25)/1000,(.25)/1000),
        5: np.arange(-.15, .15 + (.15)/1000,(.15)/1000),
        6: np.arange(-.45, .45 + (.45)/1000,(.45)/1000),
        7: np.arange(-1, 1 + (1)/1000,(1)/1000)
    }


    curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
                top_e=mean_top_e, top_d=mean_top_d, bot_e=mean_bot_e, bot_d=mean_bot_d,
                path=output_dir + f'/{s}_mean', ticks=ticks[s], l_contour=contours[s],
                cmap='bwr', cbar_label='(m/s)', cbar_ticks=contours[s],
                bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])


    # Intensidade Filtrada

    curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
                top_e=meanf_top_e, top_d=meanf_top_d, bot_e=meanf_bot_e, bot_d=meanf_bot_d,
                path=output_dir + f'/{s}_mean_filt', ticks=ticks[s], l_contour=contours[s],
                cmap='bwr', cbar_label='(m/s)', cbar_ticks=contours[s],
                bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])


    # Porcentagem

    l_contour = np.array([20, 50])
    cbar_ticks = np.arange(0, 101, 10)


    perc_top_e = (varf_top_e/var_top_e) * 100
    perc_top_d = (varf_top_d/var_top_d) * 100
    perc_bot_e = (varf_bot_e/var_bot_e) * 100
    perc_bot_d = (varf_bot_d/var_bot_d) * 100

    curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
                top_e=perc_top_e, top_d=perc_top_d, bot_e=perc_bot_e, bot_d=perc_bot_d,
                path=output_dir + f'/{s}_perc', ticks=np.arange(0,100.5,5), l_contour=l_contour,
                cmap='inferno', cbar_label='%', cbar_ticks=cbar_ticks,
                bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])


    # STD medio


    contours = {
        1 : np.round(np.arange(0, .31, .03), 2),
        2 : np.round(np.arange(0, .31, .03), 2),
        3 : np.round(np.arange(0, .31, .03), 2),
        4 : np.round(np.arange(0, .31, .03), 2),
        5 : np.round(np.arange(0, .31, .03), 2),
        6 :  np.round(np.arange(0, .31, .03), 2), 
        7 : np.round(np.arange(0, .37, .03), 2)
    }

    l_contour = contours[s]
    cbar_ticks = contours[s]



    ticks = {
        1: np.arange(0, .3 + (.3)/1000,(.3)/1000),
        2: np.arange(0, .24 + (.24)/1000,(.24)/1000),
        3: np.arange(0, .24 + (.24)/1000,(.24)/1000),
        4: np.arange(0, .21 + (.21)/1000,(.21)/1000),
        5: np.arange(0, .15 + (.15)/1000,(.15)/1000),
        6: np.arange(0, .3 + (.3)/1000,(.3)/1000),
        7: np.arange(0, .36 + (.36)/1000,(.36)/1000)
    }



    curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
                top_e=np.sqrt(var_top_e), top_d=np.sqrt(var_top_d), bot_e=np.sqrt(var_bot_e), bot_d=np.sqrt(var_bot_d),
                path=output_dir + f'/{s}_std', ticks=ticks[s], l_contour=contours[s],
                cmap='inferno', cbar_label='(m/s)', cbar_ticks=contours[s],
                bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])
    

    # STD Filtrado

    curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
                top_e=np.sqrt(varf_top_e), top_d=np.sqrt(varf_top_d), bot_e=np.sqrt(varf_bot_e), bot_d=np.sqrt(varf_bot_d),
                path=output_dir + f'/{s}_std_f', ticks=ticks[s], l_contour=contours[s],
                cmap='inferno', cbar_label='(m/s)', cbar_ticks=contours[s],
                bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])
    
    plt.close('all')

