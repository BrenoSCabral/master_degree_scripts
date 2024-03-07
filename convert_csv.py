# Script criado para converter as saidas dos modelos em csv (pontos especificos)
output_path = '/home/bcabral/'

# modelos = ['BRAN',  'CGLO',  'ECCO',  'FOAM',  'GLOR12',  'GLOR4',  'HYCOM',  'ORAS',  'SODA']
# model = 'BRAN'

# Importando bibliotecas
import os
import xarray as xr
import re
import sys

model = sys.argv[1]

# Diretorio dos modelos
dir_model = '/data3/MOVAR/modelos/REANALISES/'

# se tiver algo que nao seja numero na pasta ele nao vai pegar
regnumber = re.compile(r'\d+')


pontos_dado = {
    'Cananeia' : [-25.02, -47.93],
    'Fortaleza' : [-3.72, -38.47],
    'Ilha_Fiscal' : [-22.90, -43.17],
    'Imbituba' : [-28.13, -48.40],
    'Macae' : [-22.23, -41.47],
    'Rio_Grande' : [-32.13, -52.10],
    'Salvador' : [-12.97, -38.52],
    'Santana' : [-0.06, -51.17],
    #'São Pedro e São Paulo' : [3.83, -32.40],
    'Ubatuba' : [-23.50, -45.12],
    'Rio Grande 2' : [-32.17, -52.09], #RS
    'Tramandaí' : [-30.00, -50.13], #RS
    'Paranagua' : [-25.50, -48.53], #PR
    'Pontal do Sul' : [-25.55, -48.37], #PR
    'Ilhabela' : [-23.77, -45.35], #SP
    'DHN' : [-22.88, -43.13],#RJ
    'Ribamar': [-2.56, -44.05]#MA
    }

for year in os.listdir(f'{dir_model}{model}SSH/'):
    if not regnumber.search(year):
        continue
    reanalisys = xr.open_mfdataset(f'{dir_model}{model}SSH/{year}/*nc')

    ## precisa resolver o meio de campo aqui pra ver como faco pra pegar
    ### as variaveis
    ### ver como eu fiz p corrente p mudar o nome

    for ponto in pontos_dado:
        lat, lon = pontos_dado[ponto]
        ssh = reanalisys.sossheig.sel(y=lat, x=lon, method='nearest')
        ssh.to_dataframe().to_csv(f'{output_path}{model}_{ponto}_{year}.csv')

