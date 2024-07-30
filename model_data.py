from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import numpy as np
import os
from read_reanalisys import set_reanalisys_dims
from read_data import read_exported_series, treat_exported_series
import filtro
import json


def get_model_region(lat, lon, model_name, reanalisys, set_dims = True):
    # reanalisys = xr.open_mfdataset(model_path + '/*SSH*.nc')
    if set_dims:
        reanalisys = set_reanalisys_dims(reanalisys, model_name)

    lat_idx = np.abs(reanalisys['latitude'] - lat).argmin().item()
    lon_idx = np.abs(reanalisys['longitude'] - lon).argmin().item()

    lat_indices = range(lat_idx-3, lat_idx+2)
    lon_indices = range(lon_idx-2, lon_idx+3)
    reanal_subset = reanalisys.isel(latitude=lat_indices, longitude=lon_indices)

    return reanal_subset


def get_correlation(filt_data, model_subset, fig_folder):
    correlation = np.empty((len(model_subset.latitude), len(model_subset.longitude)))
    latlons = np.zeros((len(model_subset.latitude), len(model_subset.longitude)), dtype=object)
    for i in range(len(model_subset.latitude)):
        for j in range(len(model_subset.longitude)):
            model_series = model_subset.isel(latitude=i, longitude=j)
            mod_ssh = model_series['ssh'].values
            mod_time = model_series['time'].values
            mod_band = filtro.filtra_dados(mod_ssh, mod_time, 'band', modelo = True)
            correlation[i, j] = np.corrcoef(filt_data, mod_band)[0, 1]
            latlons[i,j] = (float(model_series['latitude'].values), float(model_series['longitude'].values))
    return correlation, latlons


def mapa_corr(lon_point, lat_point, reanal_subset, correlation, nome_modelo, nome_ponto, fig_folder):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon_point+1, lon_point-1, lat_point+1, lat_point-1], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN)
    ax.plot(lon_point, lat_point,
                color='red', linewidth=2, marker='o',
                transform=ccrs.PlateCarree()
                )  
    ax.add_feature(cfeature.LAKES, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    longitude, latitude = np.meshgrid(reanal_subset.longitude, reanal_subset.latitude)

    c = ax.pcolormesh(longitude, latitude, correlation, transform=ccrs.PlateCarree(), cmap='coolwarm')
    cb = plt.colorbar(c, orientation='horizontal', pad=0.05)
    cb.set_label('Correlation')


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

    # plt.tight_layout()

    plt.title('Corr. ' + nome_modelo + ' x ' + nome_ponto)

    os.makedirs(f'{fig_folder}/correlacao/{nome_ponto}/', exist_ok=True)
    plt.savefig(f'{fig_folder}/correlacao/{nome_ponto}/{nome_modelo}.png')


def write_json(latlon, point, model):
    {point: {model: latlon}}


def get_corr(data_name, server, year):
    # server = False
    # data_name = 'santana_2014.csv'
    # year = 2014

    if server:
        model_path = '/data3/MOVAR/modelos/REANALISES/'
        data_path = '/home/bcabral/mestrado/data/'
        fig_folder = '/home/bcabral/mestrado/fig/'

    else:
        model_path = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/BRAN/2014/'
        fig_folder = '/Users/breno/Documents/Mestrado/resultados/2012/figs'
        data_path  = '/Users/breno/Documents/Mestrado/resultados/2012/data/'

    

    data_raw = read_exported_series(data_path + str(year) + '/' + data_name)
    # data_raw = read_exported_series('/Users/breno/Documents/Mestrado/resultados/2012/data/' + data_name)
    data = treat_exported_series(data_raw)
    # data.resample('H').median()

    lat = data['lat'][0]
    lon = data['lon'][0]

    data_low = filtro.filtra_dados(data['ssh'], data.index, 'low', modelo = False)
    data_low = data_low[::24]
    data_high = filtro.filtra_dados(data_low, data.index[::24], 'high')

    data_filt = data_high

    # make_spectrum(data['ssh'], data_filt, data_name[:-4])

    # ECCO atualiza de 3 em 3 dias
    models = ['BRAN', 'CGLO', 'FOAM', 'GLOR12', 'GLOR4', 'HYCOM', 'ORAS']#  'ECCO' ,'SODA']
    json_dict = {}
    for model in models:
        print(model)
        # setando caminho do servidor
        reanalisys = xr.open_mfdataset(model_path + model + '/SSH/' + str(year)  + '/*.nc')
    #model = 'BRAN'
        reanal_subset = get_model_region(lat, lon, model, reanalisys)
        reanal_subset['ssh'].load()
        reanal_subset = reanal_subset.sel(time=slice(data.index[0], data.index[-1]))


        correlation, latlons= get_correlation(data_filt, reanal_subset, fig_folder)

        correlation[np.isnan(correlation)] = -np.inf
        max_index = np.unravel_index(np.argmax(correlation, axis=None), correlation.shape)

        json_dict[model] = latlons[max_index]
        # mapa_corr(lon, lat, reanal_subset, correlation, model, '_'.join(str.split(data_name, '_')[:-1]), fig_folder)
    
    with open(f'/home/bcabral/mestrado/{data_name[:-4]}.json', 'w') as f:
        json.dump(json_dict, f)


def main():
    print('rodando main')

    # places = ['Santana_2014.csv', 'Fortaleza_2014.csv', 'salvador2_2014.csv',
    #         'macae_2014.csv', 'ilha_fiscal_2014.csv', 'ubatuba_2014.csv', 'cananeia_2014.csv', 'imbituba_2014.csv']

    places = ['CAPITANIA DE SALVADORm_2012.csv', 'Fortaleza_2012.csv', 'ilha_fiscal_2012.csv', 'TEPORTIm_2012.csv']# , 'macae_2012.csv']

    for place in places:
        print(f'rodando para {place}')
        get_corr(place, True, 2012)

        # data_raw = read_exported_series(f'/Users/breno/Documents/Mestrado/resultados/2014/data/{place}')
        # data = treat_exported_series(data_raw)

        # plt.figure(figsize=(15,10))
        # plt.plot(data['ssh'])
        # plt.grid()
        # plt.savefig('/Users/breno/Documents/Mestrado/resultados/2014/figs/series/' + str.split(place, '.')[0] + '.png')

if __name__ == "__main__":
    main()

