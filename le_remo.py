#------------------------------BIBLIOTECAS

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# import haxs
from os.path import join
from datetime import datetime
import scipy.io as sio
# import netCDF4 as nc
import datetime
import pandas as pd
import matplotlib as mpl
#import funcao_mod as fm
from sys import path
path.append(
    '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/resultados_REMO/rotinas_leitura_pY'
)
import funcoes_mod as fm
#-------------------------------------------------------------------------------------------

# /Volumes/BRENO_HD/dados_mestrado/dados/reanalise/resultados_REMO/rotinas_leitura_pY
path_reanalise = '/Volumes/BRENO_HD/dados_mestrado/dados/reanalise/resultados_REMO/'

# haxby_lua = haxs.barcolor('haxby_lua')
# haxby = haxs.barcolor('haxby')

#def main(ano,mes,dia):   
# DEFININDO PARAMETROS DO MODELO
IDM = 601 #numeros de pontod idm
JDM = 727 #numeros de pontos jdm
KDM = 32 #numeros total de camadAs

##-------------DIRETÓRIOS

dire = join('/home/hycom_resultados/LSE36_V01/')
dire_fig= '/home/arofcn/LSE36/analises/campos_data'
dire_ostia = (f'{dire}base_de_dados/OSTIA/ostia_2009')
dire_aviso = (f'{dire}base_de_dados/AVISO_MAPS/2009')
dire_merc = (f'{dire}base_de_dados/MERCATOR')

##---------------------------------------LEITURA DO GRID LSE36
fid = (f'{path_reanalise}SSH_LSE24_V01_2014_diario')
plon,plat = fm.grid(IDM,JDM,f'{path_reanalise}regional.grid_LSE24_V01.a')

data_lat = -23.04 ## USA ESSE AQUI PRA FUNCIONAR
data_lon = -43.12
np.where(plon>=data_lon)

plon[261]

pd.DataFrame(index=date)
# fazer loop pra pegar tds os dias do ano for i in 
campo_SSH = fm.ler_campo_ex(7,IDM,JDM,fid,ssh = False)
campo_SSH[0,261] # pra pegar determinado ponto latlon

# plt.pcolor(campo_SSH,shading='auto')


ano = 2014;mes = 1;dia = 1
d = datetime.datetime(ano,mes,dia)
data_ = str(d.date())
data = data_.replace('-','')
INDEX = int(d.strftime('%j'))

archv_fid36S = (f'{path_reanalise}SSH_LSE24_V01_2014_diario') # SSh
ssh361 = fm.ler_campo_ex(INDEX,IDM,JDM,archv_fid36S,ssh = True)