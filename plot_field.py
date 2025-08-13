import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import model_filt
from read_reanalisys import set_reanalisys_dims


model = 'BRAN'
reanal = {}
years = range(2001, 2002)
for year in years:
    # reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Volumes/BRENO_HD/BRAN/' + str(year)  + '/*.nc')
    #                                     , model)
    reanal[year] = set_reanalisys_dims(xr.open_mfdataset('/Users/breno/mestrado/REANALISES_TEMP/BRAN/' + str(year)  + '/*.nc')
                                        , model)

reanalisys = xr.concat(list(reanal.values()), dim="time")


lat_range = slice(-35, 5)
lon_range = slice(-55,-33)

reanalisys = reanalisys.sel(latitude=lat_range, longitude=lon_range)



filt_val =  model_filt.filtra_reanalise(reanalisys)

reanalisys['ssh_filt'] = filt_val

bathy_file_path = '/Users/breno/mestrado/gebco_2024_n5.0_s-60.0_w-70.0_e-30.0.nc'
bathy_ds_raw = xr.open_dataset(bathy_file_path)

depth = bathy_ds_raw['elevation'].isel(time=0, drop=True) if 'time' in bathy_ds_raw.dims else bathy_ds_raw['elevation']
# interpolar a batimetria para o grid da reanálise

depth =  depth.rename({'lat': 'latitude', 'lon': 'longitude'})

depth_interp = depth.interp(
    latitude=reanalisys.latitude,
    longitude=reanalisys.longitude
)

# cria a máscara booleana: True onde profundidade >= 1000 m
mask_2d = depth_interp >= -1000

# expande a máscara para ter a dimensão "time"
mask_3d = mask_2d.expand_dims({'time': reanalisys.time})

ssh_masked = reanalisys['ssh_filt'].where(mask_3d, other=0)



import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def plot_map(i):
    cmap = plt.get_cmap('bwr').copy()   # Copy to modify safely
    cmap.set_bad(color='lightgrey')     # Color for NaNs (land mask)

    fig = plt.figure(figsize=(6, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # # Plot the SSH data
    # ssh_masked[i].plot(
    #     ax=ax,
    #     transform=ccrs.PlateCarree(),
    #     vmin=-0.2,
    #     vmax=0.2,
    #     cmap=cmap,        # use the same cmap with set_bad
    #     add_colorbar=True,
    #     add_labels=False
    # )

    # Plot the SSH data sem colorbar automática
    im = ssh_masked[i].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=-0.2,
        vmax=0.2,
        cmap=cmap,
        add_colorbar=False,
        add_labels=False
    )

    # Adiciona manualmente a colorbar com setas (extend)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SSH (m)', fontsize=9)


    # Add coastlines and countries
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgrey')

    cs = ax.contour(
        depth_interp['longitude'], depth_interp['latitude'],
        depth_interp, levels=[-1000], colors='black', linewidths=.5)
        
    # Add latitude and longitude lines


    # Linhas de latitude e longitude a cada 5°
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    # Define os intervalos de 5 em 5 graus
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 5))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 5))

    # Formatação dos rótulos
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()



def plot_map_ne(i):
    cmap = plt.get_cmap('bwr').copy()   # Copy to modify safely
    cmap.set_bad(color='lightgrey')     # Color for NaNs (land mask)

    fig = plt.figure(figsize=(6, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # # Plot the SSH data
    # ssh_masked[i].plot(
    #     ax=ax,
    #     transform=ccrs.PlateCarree(),
    #     vmin=-0.2,
    #     vmax=0.2,
    #     cmap=cmap,        # use the same cmap with set_bad
    #     add_colorbar=True,
    #     add_labels=False
    # )


    # Plot the SSH data sem colorbar automática
    im = ssh_masked[i].plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        vmin=-0.2,
        vmax=0.2,
        cmap=cmap,
        add_colorbar=False,
        add_labels=False
    )

    # Adiciona manualmente a colorbar com setas (extend)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('SSH (m)', fontsize=9)


    # Add coastlines and countries
    ax.coastlines(resolution='10m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, color='lightgrey')

    cs = ax.contour(
        depth_interp['longitude'], depth_interp['latitude'],
        depth_interp, levels=[-1000], colors='black', linewidths=.5)
        
    # Add latitude and longitude lines


    # Linhas de latitude e longitude a cada 5°
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color='gray',
        alpha=0.5,
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    # Define os intervalos de 5 em 5 graus
    gl.xlocator = mticker.FixedLocator(range(-180, 181, 5))
    gl.ylocator = mticker.FixedLocator(range(-90, 91, 5))

    # Formatação dos rótulos
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

    ax.set_extent([-40, -34, -20, -3], crs=ccrs.PlateCarree())



for i in range(len(reanalisys['ssh_filt'])):
    plot_map(i)
    
    t = reanalisys['ssh_filt'].time[i].values
    plt.title(pd.to_datetime(t).strftime('%Y-%m-%d'))
    plt.tight_layout()

    plt.savefig(f'/Users/breno/mestrado/ssh_map/fig_ctw_{i:03d}.png')
    plt.close('all')

    plot_map_ne(i)
    t = reanalisys['ssh_filt'].time[i].values
    plt.title(pd.to_datetime(t).strftime('%Y-%m-%d'))
    plt.tight_layout()

    plt.savefig(f'/Users/breno/mestrado/ssh_map_ne/fig_ctw_{i:03d}.png')
    plt.close('all')

