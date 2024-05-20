import xarray as xr
import numpy as np
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import itertools
import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os

# path_reanalise = '/Users/breno/dados_mestrado/dados/reanalise/GLOR4/'
# figs_folder = '/Users/breno/Documents/Mestrado/tese/figs/teste/'
# long1 = (302.47 + 180) % 360 - 180

##################################################################
#                           FUNCTIONS                            #
##################################################################

# BRAN: TEMPO = 365 - LAT = 951 - LON = 1401
# HYCOM: TEMPO: 365 - LAT = 1751 - LON = 1751
# GLOR4: TEMPO = 365, LAT = 401 - LON = 561

# ----> dataset.sel({time_dim: time, lat_dim: lat, lon_dim: lon}) <---

def check_fig_path(fig_folder, model_name):
    fig_folder = fig_folder + f'{model_name}/'
    if not os.path.exists(fig_folder):
      os.makedirs(fig_folder)
    return fig_folder

def read_reanalisys(year, path_reanalise):
    '''
        Opens the reanalisys for the year of interest.
        :path_reanalise: str with the path of the reanalisys
        :year: year of interest
    '''
    reanalisys = xr.open_mfdataset(f'{path_reanalise}{year}/*SSH*')

    return reanalisys

def cut_reanalisys(reanalisys, ti, tf, lat, lon):
    '''
        Makes a selection from the realanilsys domain in time and space
        :reanalisys: Dataarray
        :ti: str - initial time in the format yyyy-mm-dd
        :tf: str - final time in the format yyyy-mm-dd
        :lat: float - latitude of the poin of interest
        :lon: float - longitude of the point of interest
    '''
    variable = list(reanalisys.data_vars)[0]

    # if dimensions['lat'] == 'lat' and dimensions['lon'] == 'lon':
    #     ssh_raw = reanalisys[variable].sel(lat=lat,
    #                                 lon=lon,
    #                                 method='nearest')
    # elif dimensions['lat'] == 'latitude' and dimensions['lon'] == 'longitude':
    #     ssh_raw = reanalisys[variable].sel(latitude=lat,
    #                                 longitude=lon,
    #                                 method='nearest')
    # elif dimensions['lat'] == 'yt_ocean' and dimensions['lon'] == 'xt_ocean':
    #     ssh_raw = reanalisys[variable].sel(yt_ocean=lat,
    #                                 xt_ocean=lon,
    #                                 method='nearest')        
    # else:
    #     raise('Check latitude/longitude dimensions name.')

    ssh_raw = reanalisys[variable].sel(latitude = lat, longitude = lon)

#    ???????
    try:
        ssh = ssh_raw.sel(time=slice(ti, tf))
    except:
        ssh = ssh_raw.sel(Time=slice(ti, tf))

    return ssh

def get_reanalisys_dims(reanalisys):
        if reanalisys == 'GLOR4' or reanalisys == 'GLOR12':
            dims = ({'time':'time', 'lat':'latitude', 'lon':'longitude'})
        elif reanalisys == 'HYCOM':
            dims = ({'time':'time', 'lat':'lat', 'lon':'lon'}) 
        elif reanalisys == 'BRAN':
            dims = ({'time':'Time', 'lat':'yt_ocean', 'lon':'xt_ocean'})

        return dims

def set_reanalisys_dims(reanalisys, name):
    if name == 'HYCOM':
        reanalisys = reanalisys.rename({'lat': 'latitude', 'lon': 'longitude'})
    elif name == 'BRAN':
        reanalisys = reanalisys.rename({'yt_ocean': 'latitude', 'xt_ocean': 'longitude'})

    return reanalisys

def get_lat_lon(reanalisys):
    return (reanalisys.latitude.values, reanalisys.longitude.values)

def get_lat_lon_old(reanalisys, dimensions):
    if dimensions['lat'] == 'lat' and dimensions['lon'] == 'lon':
        return(reanalisys.lat.values, reanalisys.lon.values)
    elif dimensions['lat'] == 'latitude' and dimensions['lon'] == 'longitude':
        return(reanalisys.latitude.values, reanalisys.longitude.values)
    elif dimensions['lat'] == 'yt_ocean' and dimensions['lon'] == 'xt_ocean':
        return (reanalisys.yt_ocean.values, reanalisys.xt_ocean.values)
    else:
        raise('Check latitude/longitude dimensions name.')

def plot_domain(lats, lons, nome_modelo, figs_folder):  
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lon_max, lon_min, lat_max, lat_min = lons[-1], lons[0], lats[-1], lats[0]
    ax.set_extent([lon_max, lon_min, lat_max, lat_min], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='0.3')
    ax.add_feature(cfeature.LAKES, alpha=0.9)  
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)

    states_provinces = cfeature.NaturalEarthFeature(
            category='cultural',  name='admin_1_states_provinces_lines',
            scale='50m', facecolor='none')
    ax.add_feature(states_provinces, edgecolor='black', zorder=10)   

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'red'}
    ax.set_title('Domínio do ' + nome_modelo)

    plt.savefig(figs_folder + f'{nome_modelo}_dominio.png')
    plt.close()

    # plt.show()

def plot_point(lat, lon, nome_modelo, nome_ponto, figs_folder, lons, lats):
    plot_point_broad(lat, lon, nome_modelo, nome_ponto, figs_folder, lons, lats)
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lon_max, lon_min, lat_max, lat_min = lon + 0.5 , lon - 0.5, lat + 0.5, lat - 0.5
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
    plt.text(lon - 0.005, lat - 0.005, nome_ponto,
          horizontalalignment='right', color = 'red', weight = 'bold',
          transform=ccrs.PlateCarree())
    plt.savefig(figs_folder + f'{nome_modelo}_{nome_ponto}.png')

def plot_point_broad(lat, lon, nome_modelo, nome_ponto, figs_folder, lons, lats):
    proj = ccrs.PlateCarree()

    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(12,12))

    lon_max, lon_min, lat_max, lat_min = lon + 20 , lon - 20, lat + 15, lat - 15
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

def roda_tudo(year, lat, lon, model_name, point_name, path_reanalise, figs_folder,
              ti=None, tf=None):
    
    figs_folder = check_fig_path(figs_folder, model_name)
    check_fig_path(figs_folder, model_name)
    
    reanalisys = read_reanalisys(str(year), path_reanalise)

    # if model_name == 'BRAN':
    #     dimensions = {'time': 'Time', 'lat' : 'yt_ocean', 'lon' : 'xt_ocean'}
    # else:
    # dimensions = get_reanalisys_dims(model_name)
    reanalisys = set_reanalisys_dims(reanalisys, model_name)

    lats, lons = get_lat_lon(reanalisys)

    plot_domain(lats, lons, model_name, figs_folder)
    plot_point(lat, lon, model_name, point_name, figs_folder, lons, lats)
    plt.close('all')

    if ti == None:
        ti = reanalisys[dimensions['time']].values[0]
        # ti = reanalisys.time.values[0]
    if tf == None:
        tf = reanalisys[dimensions['time']].values[-1]
        # tf = reanalisys.time.values[-1]

    reanalisys_point = cut_reanalisys(reanalisys, ti, tf, lat, lon)
    return (reanalisys_point)


def read_reanalisys_curr(year, path_reanalise):
    '''
        Opens the reanalisys for the year of interest.
        :path_reanalise: str with the path of the reanalisys
        :year: year of interest
    '''
    reanalisys = xr.open_mfdataset(f'{path_reanalise}{year}/*UV*')

    return reanalisys

def roda_tudo_curr(year, lat, lon, model_name, path_reanalise, figs_folder):
    figs_folder = check_fig_path(figs_folder, model_name)
    check_fig_path(figs_folder, model_name)
    
    reanalisys = read_reanalisys_curr(str(year), path_reanalise)