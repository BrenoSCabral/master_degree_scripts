import os
import numpy as np
import xarray as xr
import netCDF4 as nc
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import cmocean as cmo

# Escolhendo o modelo

modelo = input('Choose a reanalysis option: BRAN, ECCO, REMO, HYCOM, SODA, CGLO, FOAM, GLOR12, GLOR4, ORAS: ')

# Escolhendo a latitude para corte
lat_corte = input('Latitude = ')


# Definindo paths
path = '/data/MOVAR/modelos/REANALISES/'+ modelo +'/'
path_save = '/home/paulamarangoni/Documentos/Doutorado/analises_modelos/medias/'

medias = xr.open_dataset(path_save+'medias_latitude'+lat_corte+'_modelo_'+modelo+'.nc')
lon =     medias['lon']
z   =     medias['depth']
media_t = medias['mean_temperature']
media_s = medias['mean_salinity']
media_vel = medias['mean_velocity']

############################################################################################
####################################### TEMPERATURA ########################################
############################################################################################

############################
#----- Plote normal ----####
############################

# ---- Plote da temperatura média na latitude de 34 graus ------------
plt.figure(figsize=(8,7))

ticks = np.arange(np.round(media_t[0,:,:].min(),0), np.round(media_t[0,:,:].max(),0), 2)
# add isoterma de 5
ticks = np.append([1.5,5],ticks)
#ordenando
ticks.sort()

# Criando listas 
linestyles_list = ['dashed']*ticks.shape[0]
linewidths_list = [0.5]*ticks.shape[0]
# substituindo a isoterma 5 e 18, 2 e 6 , 1.5 e 4,
linestyles_list = np.insert(linestyles_list, [1,2,3,5,11],'solid')
linewidths_list = np.insert(linewidths_list, [1,2,3,5,11], 0.8)

im = plt.contourf(lon, -z, media_t[0,:,:], cmap='cmo.thermal', levels=ticks)
plt.gca().set_facecolor([0,0,0,0.6]) 
plt.colorbar(im, label='Mean Temperature (°C)', extend='both', 
             orientation='vertical', pad=0.02, fraction=0.05)

cs = plt.contour(lon, -z, media_t[0,:,:], colors='black', levels=ticks, linestyles = linestyles_list,linewidths=linewidths_list )
plt.clabel(cs, inline=True,fontsize=10)

plt.xlim(-55, 22)
plt.xlabel("Longitude (°)")
plt.ylabel("Depth (m)")

plt.annotate("South America", xy=(-53, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))
plt.annotate("Africa", xy=(19, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))

plt.title('Mean Temperature', fontsize=10, loc='left')
plt.title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
plt.title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')

plt.savefig(path_save+'Mean_temperature_'+modelo+'_lat_'+lat_corte[1:3]+'S.png',bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()


############################
#----- 2 subplotes ----#####
############################

# ---- Plote da temperatura média na latitude de 34 graus CAMADA SUPERIOR, CAMADA INFERIOR ------------
fig, ax = plt.subplots(2,1, figsize=(8,7))

# Camada superior
media_t_sup = media_t.sel(depth=slice(0,400))
z_sup = z.sel(depth=slice(0,400))

#ticks1 = np.arange(8.0, 22+2, 2)
ticks1 = np.arange(np.round(media_t_sup[0,:,:].min(),0), np.round(media_t_sup[0,:,:].max(),0), 2)

im1 = ax[0].contourf(lon, -z_sup, media_t_sup[0,:,:], cmap='cmo.thermal', levels=ticks1)
ax[0].set_facecolor([0,0,0,0.6]) 
plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0])

cs1 = ax[0].contour(lon, -z_sup, media_t_sup[0,:,:], colors='black', levels=ticks1, linestyles= 'dashed',linewidths=0.5 )
ax[0].clabel(cs1, inline=True,fontsize=10)

ax[0].set_xlim(-55, 22)
ax[0].set_xlabel("Longitude (°)")
ax[0].set_ylabel("Depth (m)")

ax[0].set_title('Mean Temperature: 0 - 400m', fontsize=10, loc='left')
ax[0].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
ax[0].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada inferior
media_t_inf = media_t.sel(depth=slice(400,5727.917))
z_inf = z.sel(depth=slice(400,5727.917))

#ticks2 = np.arange(-2,12+2, 2)
ticks2 = np.arange(np.round(media_t_inf[0,:,:].min(),0), np.round(media_t_inf[0,:,:].max(),0), 2)

im2 = ax[1].contourf(lon, -z_inf, media_t_inf[0,:,:], cmap='cmo.thermal', levels=ticks2)
ax[1].set_facecolor([0,0,0,0.6]) 
plt.colorbar(im2, label='Mean Temperature (°C)', extend='both', 
             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[1])

cs2 = ax[1].contour(lon, -z_inf, media_t_inf[0,:,:], colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
ax[1].clabel(cs2, inline=True,fontsize=10)

ax[1].set_xlim(-55, 22)
ax[1].set_xlabel("Longitude (°)")
ax[1].set_ylabel("Depth (m)")


ax[1].annotate("South America", xy=(-53, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))
ax[1].annotate("Africa", xy=(19, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))

ax[1].set_title('Mean Temperature: 400 - 6000m', fontsize=10, loc='left')
#ax[1].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[1].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')

plt.tight_layout()

plt.savefig(path_save+'Mean_temperature_'+modelo+'_lat_'+lat_corte[1:3]+'S_camadas.png',bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()


############################
#----- 6 subplotes ----#####
############################

# ---- Plote da temperatura média na latitude de 34 graus CAMADA SUPERIOR, CAMADA INFERIOR ------------
fig, ax = plt.subplots(2,3, figsize=(12,9))

# Camada superior (esquerda)---------------------------------------------------
media_t_sup00 = media_t.sel(depth=slice(0,400),lon=slice(-55,-45))
z_sup00 = z.sel(depth=slice(0,400))
lon00 = lon.sel(lon=slice(-55,-45))

#ticks1 = np.arange(np.round(media_t_sup00[0,:,:].min(),0), np.round(media_t_sup00[0,:,:].max(),0), 2)
ticks1 = np.arange(8,22+2, 2)

im1 = ax[0,0].contourf(lon00, -z_sup00, media_t_sup00[0,:,:], cmap='cmo.thermal', levels=ticks1)
ax[0,0].set_facecolor([0,0,0,0.6]) 
#plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
#             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,0])

cs1 = ax[0,0].contour(lon00, -z_sup00, media_t_sup00[0,:,:], colors='black', levels=ticks1, linestyles= 'dashed',linewidths=0.5 )
ax[0,0].clabel(cs1, inline=True,fontsize=10)

ax[0,0].set_xlim(-55, -45)
#ax[0,0].set_xlabel("Longitude (°)")
ax[0,0].set_ylabel("Depth (m)")
ax[0,0].set_xticks([])

ax[0,0].set_title('Mean Temperature: 0 - 400m', fontsize=10, loc='left')
#ax[0,0].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[0,0].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada superior (meio)---------------------------------------------------
media_t_sup01 = media_t.sel(depth=slice(0,400),lon=slice(-45,10))
z_sup01 = z.sel(depth=slice(0,400))
lon01 = lon.sel(lon=slice(-45,10))

#ticks1 = np.arange(np.round(media_t_sup01[0,:,:].min(),0), np.round(media_t_sup01[0,:,:].max(),0), 2)
ticks1 = np.arange(8,22+2, 2)

im1 = ax[0,1].contourf(lon01, -z_sup01, media_t_sup01[0,:,:], cmap='cmo.thermal', levels=ticks1)
ax[0,1].set_facecolor([0,0,0,0.6]) 
#plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
#             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,1])

cs1 = ax[0,1].contour(lon01, -z_sup01, media_t_sup01[0,:,:], colors='black', levels=ticks1, linestyles= 'dashed',linewidths=0.5 )
ax[0,1].clabel(cs1, inline=True,fontsize=10)

ax[0,1].set_xlim(-45, 10)
ax[0,1].set_yticks([])
ax[0,1].set_xticks([])
#ax[0,1].set_xlabel("Longitude (°)")
#ax[0,1].set_ylabel("Depth (m)")

#ax[0,1].set_title('Mean Temperature: 0 - 400m', fontsize=10, loc='left')
ax[0,1].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[0,1].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada superior (direita)---------------------------------------------------
media_t_sup02 = media_t.sel(depth=slice(0,400),lon=slice(10,22))
z_sup02 = z.sel(depth=slice(0,400))
lon02 = lon.sel(lon=slice(10,22))


#ticks1 = np.arange(np.round(media_t_sup02[0,:,:].min(),0), np.round(media_t_sup02[0,:,:].max(),0), 2)
ticks1 = np.arange(8,22+2, 2)

im1 = ax[0,2].contourf(lon02, -z_sup02, media_t_sup02[0,:,:], cmap='cmo.thermal', levels=ticks1)
ax[0,2].set_facecolor([0,0,0,0.6]) 
plt.colorbar(im1, label='Mean Temperature (°C)', extend='both', 
             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[0,2])

cs1 = ax[0,2].contour(lon02, -z_sup02, media_t_sup02[0,:,:], colors='black', levels=ticks1, linestyles= 'dashed',linewidths=0.5 )
ax[0,2].clabel(cs1, inline=True,fontsize=10)

ax[0,2].set_xlim(10, 22)
ax[0,2].set_yticks([])
ax[0,2].set_xticks([])
#ax[0,2].set_xlabel("Longitude (°)")
#ax[0,2].set_ylabel("Depth (m)")

#ax[0,2].set_title('Mean Temperature: 0 - 400m', fontsize=10, loc='left')
#ax[0,2].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[0,2].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada inferior (esquerda)----------------------------------------------------------------
media_t_inf10 = media_t.sel(depth=slice(400,5727.917),lon=slice(-55,-45))
z_inf10 = z.sel(depth=slice(400,5727.917))
lon10 = lon.sel(lon=slice(-55,-45))

#ticks2 = np.arange(np.round(media_t_inf10[0,:,:].min(),0), np.round(media_t_inf10[0,:,:].max(),0), 2)
ticks2 = np.arange(0,12+2, 2)

im2 = ax[1,0].contourf(lon10, -z_inf10, media_t_inf10[0,:,:], cmap='cmo.thermal', levels=ticks2)
ax[1,0].set_facecolor([0,0,0,0.6]) 
#plt.colorbar(im2, label='Mean Temperature (°C)', extend='both', 
#             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[1,0])

cs2 = ax[1,0].contour(lon10, -z_inf10, media_t_inf10[0,:,:], colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
ax[1,0].clabel(cs2, inline=True,fontsize=10)

ax[1,0].set_xlim(-55,-45)
#ax[1,0].set_xlabel("Longitude (°)")
ax[1,0].set_ylabel("Depth (m)")


ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))
#ax[1,0].annotate("Africa", xy=(19, -4000), rotation=90, 
#            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
#            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))

ax[1,0].set_title('Mean Temperature: 400 - 6000m', fontsize=10, loc='left')
#ax[1].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[1].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada inferior (meio)----------------------------------------------------------------
media_t_inf11 = media_t.sel(depth=slice(400,5727.917),lon=slice(-45,10))
z_inf11 = z.sel(depth=slice(400,5727.917))
lon11 = lon.sel(lon=slice(-45,10))

#ticks2 = np.arange(np.round(media_t_inf11[0,:,:].min(),0), np.round(media_t_inf11[0,:,:].max(),0), 2)
ticks2 = np.arange(0,12+2, 2)

im2 = ax[1,1].contourf(lon11, -z_inf11, media_t_inf11[0,:,:], cmap='cmo.thermal', levels=ticks2)
ax[1,1].set_facecolor([0,0,0,0.6]) 
#plt.colorbar(im2, label='Mean Temperature (°C)', extend='both', 
#             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[1,1])

cs2 = ax[1,1].contour(lon11, -z_inf11, media_t_inf11[0,:,:], colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
ax[1,1].clabel(cs2, inline=True,fontsize=10)

ax[1,1].set_xlim(-45,10)
ax[1,1].set_yticks([])
ax[1,1].set_xlabel("Longitude (°)")
#ax[1,1].set_ylabel("Depth (m)")


#ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
#            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
#            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))
#ax[1,0].annotate("Africa", xy=(19, -4000), rotation=90, 
#            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
#            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))

#ax[1,1].set_title('Mean Temperature: 400 - 6000m', fontsize=10, loc='left')
#ax[1].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[1].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')


# Camada inferior (meio)----------------------------------------------------------------
media_t_inf12 = media_t.sel(depth=slice(400,5727.917),lon=slice(10,22))
z_inf12 = z.sel(depth=slice(400,5727.917))
lon12 = lon.sel(lon=slice(10,22))

#ticks2 = np.arange(np.round(media_t_inf12[0,:,:].min(),0), np.round(media_t_inf12[0,:,:].max(),0), 2)
ticks2 = np.arange(0,12+2, 2)

im2 = ax[1,2].contourf(lon12, -z_inf12, media_t_inf12[0,:,:], cmap='cmo.thermal', levels=ticks2)
ax[1,2].set_facecolor([0,0,0,0.6]) 
plt.colorbar(im2, label='Mean Temperature (°C)', extend='both', 
             orientation='vertical', pad=0.02, fraction=0.05, ax = ax[1,2])

cs2 = ax[1,2].contour(lon12, -z_inf12, media_t_inf12[0,:,:], colors='black', levels=ticks2, linestyles= 'dashed',linewidths=0.5 )
ax[1,2].clabel(cs2, inline=True,fontsize=10)

ax[1,2].set_xlim(10,22)
ax[1,2].set_yticks([])
#ax[1,2].set_xlabel("Longitude (°)")
#ax[1,2].set_ylabel("Depth (m)")


#ax[1,0].annotate("South America", xy=(-53, -4000), rotation=90, 
#            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
#            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))
ax[1,2].annotate("Africa", xy=(19, -4000), rotation=90, 
            fontsize=11, fontweight='bold', color='black', ha='center',va='center',
            bbox=dict(boxstyle="round,pad=0.2",fc="w", alpha=0.7))

#ax[1,2].set_title('Mean Temperature: 400 - 6000m', fontsize=10, loc='left')
#ax[1,2].set_title('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
#ax[1,2].set_title('Reanalysis: '+modelo, fontsize=10, loc='right', fontweight='bold')

plt.tight_layout()
#plt.suptitle('Latitude: '+lat_corte+'°', fontsize=10, loc='center')
plt.suptitle('Reanalysis: '+modelo, fontsize=10, fontweight='bold',x=0.85)

plt.savefig(path_save+'Mean_temperature_'+modelo+'_lat_'+lat_corte[1:3]+'S_6camadas.png',bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()