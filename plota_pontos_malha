home_path = '/Users/breno/Documents/Mestrado/estudos_modelos/'
# home_path =  '/home/bcabral/'

import xarray as xr
import matplotlib.ticker as mticker
import os

# from le_reanalise import get_lat_lon, set_reanalisys_dims
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude'})

    return reanalisys


def get_lat_lon(reanalisys):
    return (reanalisys.latitude.values, reanalisys.longitude.values)


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


pontos_dado = {
    'cananeia' : [-25.02, -47.93],
    'fortaleza' : [-3.72, -38.47],
    'ilha_fiscal' : [-22.90, -43.17],
    'imbituba' : [-28.13, -48.40],
    'macae' : [-22.23, -41.47],
    'rio_grande' : [-32.13, -52.10],
    'salvador' : [-12.97, -38.52],
    # 'santana' : [-0.06, -51.17],
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

pontos_modelo = {
    'BRAN':{
        'cananeia' : [-25.05, -47.85],
        'fortaleza' : [-3.65, -38.45],
        'ilha_fiscal' : [-23.05, -43.15],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.25, -41.45],
        'rio_grande' : [-32.25, -52.05],
        'salvador' : [-13.15, -38.55],
        'ubatuba' : [-23.55, -45.05],
        'rio_grande2' : [-32.25, -52.05], #RS
        'tramandai' : [-30.05, -50.05], #RS
        'paranagua' : [-25.65, -48.35], #PR
        'pontal_sul' : [-25.65, -48.35], #PR
        'ilhabela' : [-23.65, -45.25], #SP
        'dhn' : [-23.05, -43.15],#RJ
        'ribamar': [-2.25, -44.05]#MA}
    },
    'CGLO':{
        'cananeia' : [-25, -47.75],
        'fortaleza' : [-3.5, -38.5],
        'ilha_fiscal' : [-23, -43],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.25, -41.5],
        'rio_grande' : [-32.25, -52],
        'salvador' : [-13.25, -38.5],
        # 'santana' : [-0.06, -51.17], # nao usar!
        #'São Pedro e São Paulo' : [3.83, -32.40],
        'ubatuba' : [-23.5, -45],
        'rio_grande2' : [-32.25, -52], #RS
        'tramandai' : [-30, -50], #RS
        'paranagua' : [-25.75, -48.25], #PR
        'pontal_sul' : [-24, -45.5], #PR
        'ilhabela' : [-23.65, -45.25], #SP
        'dhn' : [-23.25, -43.25],#RJ
        'ribamar': [-2.25, -44]#MA}
    },
    'ECCO':{
        'cananeia' : [-25.125, -47.875],
        'fortaleza' : [-3.625, -38.375],
        'ilha_fiscal' : [-23.125, -43.125],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.375, -41.375],
        'rio_grande' : [-32.375, -51.875],
        'salvador' : [-13.125, -38.375],
        # 'santana' : [-0.06, -51.17], # nao usar!
        #'São Pedro e São Paulo' : [3.83, -32.40],
        'ubatuba' : [-23.625, -44.875],
        'rio_grande2' : [-32.375, -51.875], #RS
        'tramandai' : [-30.125, -50.125], #RS
        'paranagua' : [-25.625, -48.125], #PR
        'pontal_sul' : [-25.625, -48.125], #PR
        'ilhabela' : [-24.125, -45.375], #SP
        'dhn' : [-23.125, -43.125],#RJ
        'ribamar': [-2.125, -44.125]#MA}
    },
    'GLOR12':{
        'cananeia' : [-25.0833, -47.833],
        'fortaleza' : [-3.66, -38.3925 - 0.08],
        'ilha_fiscal' : [-23, -43.1],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.25, -41.43-0.08],
        'rio_grande' : [-32.25, -52.08],
        'salvador' : [-13.125, -38.50],
        'ubatuba' : [-23.58, -45.08],
        'rio_grande2' : [-32.25, -52.08], #RS
        'tramandai' : [-30, -50.08], #RS
        'paranagua' : [-25.58, -48.20], #PR
        'pontal_sul' : [-25.66, -48.33], #PR
        'ilhabela' : [-23.66, -45.33], #SP
        'dhn' : [-23, -43.08],#RJ
        'ribamar': [-2.25, -44]#MA}
    },
    'HYCOM':{
        'cananeia' : [-25.04, -47.84],
        'fortaleza' : [-3.68, -38.4 - 0.08],
        'ilha_fiscal' : [-23.04, -43.12],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.24, -41.44],
        'rio_grande' : [-32.24, -52.08],
        'salvador' : [-13.04, -38.48],
        # 'santana' : [-0.06, -51.17], # nao usar!
        #'São Pedro e São Paulo' : [3.83, -32.40],
        'ubatuba' : [-23.6, -45.12],
        'rio_grande2' : [-32.24, -52.08], #RS
        'tramandai' : [-30, -50.08], #RS
        'paranagua' : [-25.6, -48.24], #PR
        'pontal_sul' : [-25.6, -48.24], #PR
        'ilhabela' : [-23.68, -45.2], #SP
        'dhn' : [-23.04, -43.12],#RJ
        'ribamar': [-2.32, -44.08]#MA}
    },
    'ORAS':{
        'cananeia' : [-25, -47.75],
        'fortaleza' : [-3.75, -38.25],
        'ilha_fiscal' : [-23, -43],
        'imbituba' : [-28.13, -48.40], # latlon dele ta esquisita, vou deixar como ta
        'macae' : [-22.25, -41.5],
        'rio_grande' : [-32.25, -52],
        'salvador' : [-13.25, -38.5],
        # 'santana' : [-0.06, -51.17], # nao usar!
        #'São Pedro e São Paulo' : [3.83, -32.40],
        'ubatuba' : [-23.5, -45],
        'rio_grande2' : [-32.25, -52], #RS
        'tramandai' : [-30, -50], #RS
        'paranagua' : [-25.75, -48.25], #PR
        'pontal_sul' : [-25.75, -48.25], #PR
        'ilhabela' : [-23.75, -45], #SP
        'dhn' : [-23, -43],#RJ
        'ribamar': [-2.25, -44]#MA}
    }
    }

pontos_modelo['FOAM'] = pontos_modelo['CGLO']
pontos_modelo['GLOR4'] = pontos_modelo['CGLO']

modelos = ['BRAN', 'CGLO', 'ECCO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  ,'SODA']


for model in modelos:
    path_model = f'/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/{model}/2014/{model}_SSH_2014-08-31.nc'

# dir_model = '/data3/MOVAR/modelos/REANALISES/'
# for model in os.listdir(dir_model):
#     os.listdir(f'{dir_model}{model}/SSH/')
#     ## essa eh uma logica que eu preciso trabalhar mas que nao da pra ver agr com o server fora do ar.

    reanalisys = xr.open_dataset(path_model)
    reanalisys = set_reanalisys_dims(reanalisys, model)
    lats, lons = get_lat_lon(reanalisys)

    for ponto in pontos_dado:
        lat, lon = pontos_dado[ponto]
        lat_set_mod, lon_set_mod = pontos_modelo[model][ponto]

        reanalisys_point = reanalisys.sel(latitude=lat_set_mod, longitude=lon_set_mod,
                                     method='nearest')
        
        lat_mod = float(reanalisys_point.latitude.values)
        lon_mod = float(reanalisys_point.longitude.values)
        plot_grid_point(lat, lon, model, ponto, f'{home_path}{ponto}/',
                lons, lats, lat_mod, lon_mod)