import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from plota_dados_marinha import get_lat_lon, nome_estacao

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from sys import path
import csv

def find_point(name):
    pontos = {
        'cananeia' : [-25.02, -47.93],
        'fortaleza' : [-3.72, -38.47],
        'ilha_fiscal' : [-22.90, -43.17],
        'imbituba' : [-28.13, -48.40],
        'macae' : [-22.23, -41.47],
        'rio_grande' : [-32.13, -52.10],
        'salvador' : [-12.97, -38.52],
        'santana' : [-0.06, -51.17],
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
    
    for ponto in pontos:
        if ponto.lower() in name.lower():
            return pontos[ponto]


def read_gloss(path, name):
    ponto = find_point(name)
    serie = pd.read_csv(path)

    if len(serie.columns) == 5:
        serie.loc[len(serie)] = serie.columns.to_list()
        serie.columns=['ano', 'mes', 'dia', 'hora', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:00:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]
        
    elif len(serie.columns) == 6:
        serie.columns=['ano', 'mes', 'dia', 'hora', 'minuto', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['ano'][i]
            mes = serie['mes'][i]
            dia = serie['dia'][i]
            hora = serie ['hora'][i]
            minu = serie ['minuto'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:00')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['ano', 'mes', 'dia', 'hora', 'minuto'], inplace=True)

        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]

    elif len(serie.columns) == 7:
        # serie.loc[len(serie)] = serie.columns.to_list()
        serie.columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg', 'ssh']

        d_index = []
        for i in range(len(serie)):
            ano = serie['yyyy'][i]
            mes = serie[' mm'][i]
            dia = serie[' dd'][i]
            hora = serie [' hour'][i]
            minu = serie[' min'][i]
            seg = serie[' seg'][i]
            date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:{seg}')
            d_index.append(date)

        serie['data'] = d_index
        serie.set_index('data', inplace=True)
        serie.drop(columns=['yyyy', ' mm', ' dd', ' hour', ' min', ' seg'], inplace=True)


        serie = serie.sort_index()
        serie['lat'] = ponto[0]
        serie['lon'] = ponto[1]


    serie['ssh'] = serie['ssh'].astype(float)/10
    serie = serie.mask(serie < -1000)
    return serie


def read_marinha(name):
    path = f'/Users/breno/Downloads/Dados/{name}'
    
    df = pd.read_csv(path, skiprows=11, sep=';',encoding='latin-1')
    df.loc[len(df)] = df.columns.to_list()
    df.columns = ['data', 'ssh', 'Unnamed: 2']
    df = df.drop(columns=['Unnamed: 2'])

    # Converter 'data' para formato datetime, se necessário
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M')
    df.set_index('data', inplace=True)

    df['ssh'] = df['ssh'].astype(int)

    df = df.sort_index()

    lat_lon = get_lat_lon(name)
    df['lat'] = lat_lon[0]
    df['lon'] = lat_lon[1]


    return df


def get_simcosta_row(n, name):
    df = pd.read_csv(f'/Users/breno/Documents/Mestrado/dados/simcosta/{name}', skiprows=n)
    return df


def read_simcosta(name):
    n = 10
    while n < 35:
        try:
            df = get_simcosta_row(n, name)
            if df.columns[0].upper() == 'YEAR':
                break
            n+=1

        except Exception as e:
            n += 1

    d_index = []
    for i in range(len(df)):
        ano = df['YEAR'][i]
        mes = df['MONTH'][i]
        dia = df['DAY'][i]
        hora = df ['HOUR'][i]
        minu = df['MINUTE'][i]
        seg = df['SECOND'][i]
        date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:{seg}')
        d_index.append(date)

    df['data'] = d_index
    df.set_index('data', inplace=True)
    df.drop(columns=['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND'], inplace=True)


    df = df.sort_index()

    # pegalatlon
    with open(f'/Users/breno/Documents/Mestrado/dados/simcosta/{name}', 'r') as arquivo:
        # Cria um leitor CSV
        leitor_csv = csv.reader(arquivo)
        
        next(leitor_csv)
        # Lê apenas a primeira linha
        lat = next(leitor_csv)
        lon = next(leitor_csv)
        
    df['lat'] = float(lat[0][12:])
    df['lon'] = float(lon[0][12:])

    return df


def plota_pontos(pts, namefile):
    coord = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    ax.set_extent([-60, -30, 5, -40], crs=ccrs.PlateCarree())

    # ax.set_extent([-40, 10, -50, -30], crs=coord)

    for point in pts:
        marker = 'P'
        if point[2] == 's':
            color = 'royalblue'
        elif point[2] == 'm':
            color = 'lime'
        elif point[2] == '1':
            color = 'r'
            marker = 'o'
        elif point[2] == '2':
            color = 'yellow'
            
        plt.plot(point[1], point[0],
                color=color, linewidth=2, marker=marker,
                transform=ccrs.PlateCarree()
                )  
    lons = np.arange(-70, -20, 5)
    lats = np.arange(-35, 10, 5)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.ylabels_right=True
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    legend_elements = [
        Line2D([0],[0], color = 'royalblue', label='SiMCosta', marker ='P', markerfacecolor='royalblue',
               markersize=15, linestyle='None'),
        Line2D([0],[0], color = 'lime', label='Marinha', marker ='P', markerfacecolor='lime',
               markersize=15, linestyle='None'),
        Line2D([0],[0], color = 'yellow', label='GOOS', marker ='P', markerfacecolor='yellow',
               markersize=15, linestyle='None'),
        Line2D([0],[0], color = 'r', label='GLOSS', marker ='o', markerfacecolor='r',
               markersize=15, linestyle='None')                
    ]

    ax.legend(handles = legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/estudo/pontos_{namefile}.png')


def plota_series(series):
    fig, ax = plt.subplots(figsize=(10, 15))
    for serie in series:
        datas = serie[3]
        lat = serie[0]
        origem = serie[2]

        if datas[-1] < pd.Timestamp("19950101"):
            continue
        elif datas[0] < pd.Timestamp("19950101"):
            data_i = pd.Timestamp("19950101")
        else:
            data_i = datas[0]


        if origem == 's':
            color = 'royalblue'
        elif origem == 'm':
            color = 'lime'
        elif origem == '1':
            color = 'r'
        elif origem == '2':
            color = 'yellow'



        ax.hlines(y=lat, xmin=data_i, xmax=datas[-1], color=color)
        # ax.text(data_i, lat, nome, ha='right', va='center', size='small')
    
    plt.xlim([pd.Timestamp("19960101"), pd.Timestamp("20250101")])

    legend_elements = [
        Line2D([0],[0], color = 'royalblue', label='SiMCosta', markerfacecolor='royalblue', markersize=15),
        Line2D([0],[0], color = 'lime', label='Marinha', markerfacecolor='lime', markersize=15),
        Line2D([0],[0], color = 'yellow', label='GOOS', markerfacecolor='yellow', markersize=15),
        Line2D([0],[0], color = 'r', label='GLOSS', markerfacecolor='r', markersize=15)                
    ]

    ax.legend(handles = legend_elements, loc='upper left')
    plt.grid()

    plt.tight_layout()
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/estudo/abrangencia_series.png')


def recorta_infos(infos, t0, tf=pd.Timestamp("20250101")):
    new_infos = []
    for info in infos:
        data_ini = info[3][0]
        data_fim = info[3][-1]
        if  data_ini > tf or data_fim < t0:
            continue
        else:
            new_infos.append(info)

    return new_infos


def plota_recorte(recorte, t0, tf):
    for i in recorte:
        if i[2] == 's':
            color = 'royalblue'
        elif i[2] == 'm':
            color = 'lime'
        elif i[2] == '1':
            color = 'r'
        elif i[2] == '2':
            color = 'yellow'
        plt.plot(i[3], [i[0]]*len(i[3]), color=color)

    plt.xlim(t0, tf)
    # plt.legend(loc='lower left', ncol=2)
    # plt.legend(bbox_to_anchor=(1.05, 0.5))
    plt.grid(axis='x')
    plt.savefig(f'/Users/breno/Documents/Mestrado/dados/estudo/abrangencia_{t0.year}_{tf.year}.png')


def main():
    files = {}

    for i in os.listdir('/Volumes/BRENO_HD/GLOSS'):
        if i[-3:] != 'csv' or i == 'PedroPaulo_rocks_UHSLC.csv' or i[0] =='.':
            continue
        files[i+'g1'] = read_gloss('/Volumes/BRENO_HD/GLOSS/'+ i, i)

    for j in os.listdir('/Users/breno/Documents/Mestrado/dados/gloos/goos'):
        files[j+'g2'] = read_gloss('/Users/breno/Documents/Mestrado/dados/gloos/goos/'+ j, j)

    # resolucao diaria
    # for k in os.listdir('/Users/breno/Documents/Mestrado/dados/gloos/havai'):
    #     files[k+'g3'] = read_gloss('/Users/breno/Documents/Mestrado/dados/gloos/havai/'+ k, k)

    for l in os.listdir('/Users/breno/Downloads/Dados'):
        files[nome_estacao(l)[:-1]+'m'] = read_marinha(l)

    for m in os.listdir('/Users/breno/Documents/Mestrado/dados/simcosta'):
        if m[-3:] != 'csv':
            continue
            # imbituba
            # guaratuba
        files[m+'s'] = read_simcosta(m)


    infos = []
    pts = []
    for file in files:
        d = files[file]
        infos.append((d['lat'][0], d['lon'][0], file[-1], d.index, file))
        # infos.append((d.index, d['lat'][0], file.split('.')[0], file[-1]))
        # pts.append([d['lat'][0], d['lon'][0], file[-1]])

    plota_pontos(infos, 'total')

    dps95 = recorta_infos(infos, pd.Timestamp("19950101"))
    plota_pontos(dps95, '_pos1995')

    e0910 = recorta_infos(infos, pd.Timestamp("20090101"),  pd.Timestamp("20100101"))
    plota_pontos(e0910, '_2009_2010')

    e1415 = recorta_infos(infos, pd.Timestamp("20140101"),  pd.Timestamp("20150101"))
    plota_pontos(e1415, '_2014_2015')

    e1516 = recorta_infos(infos, pd.Timestamp("20150101"),  pd.Timestamp("20160101"))
    plota_pontos(e1516, '_2015_2016')
    # -12,97 nao usar salvador_glossbrasil.csvg1 -> salvador2.csvg1 e salvador_2004_2015.csvg1 sao iguais (nesse intervalo)
    # -> salvador2.csvg2/'Salvador_glossbrasil.csvg1' tem dado de 5 em 5 min e eh meio diferente a forma que o dado é mostrado 
    # -> 'CAPITANIA DE SALVADORm' eh parecido c os dois acima, mas com resultado horario e valores absolutos diferentes
    # -23,5 -> ['ubatuba.csvg1', 'Ubatuba_gloss.csvg1', 'ubatuba.csvg2'] -> os 3 sao iguais
    # -25,02 -> qqr cananeia serve
    # -3,72 -> os dois fortaleza tao diferente e tem resolucao horaria diferente. n sei oq usar.

    e1920 = recorta_infos(infos, pd.Timestamp("20190101"),  pd.Timestamp("20200101"))
    plota_pontos(e1920, '_2019_2020')

    e2224 = recorta_infos(infos, pd.Timestamp("20220101"),  pd.Timestamp("20240101"))
    plota_pontos(e2224, '_2022_2024')


def test_series(serie, lat):
    j = []
    for i in serie:
        if round(i[0], 1) == lat:
            j.append(i[-1])

    for i in range(len(j)):
        if i == 0:
            continue
        print((files[j[i]]['ssh']['2015-01-01':'2016-01-01'] != files[j[i-1]]['ssh']['2015-01-01':'2016-01-01']).sum())

'''
vou usar a serie 1516 e os seguintes arquivos ao longo da costa:
-0.06 -> Santana.csvg1
-3.72 -> Fortaleza.csvg1
-12.97 -> salvador2.csvg1
-22.23 -> macae.csvg2
-22.9 -> ilha_fiscal.csvg1
-23.5 -> ubatuba.csvg1
-25.02 -> cananeia.csvg2
-26.23 -> Porto de são francisco do sul (marinha)
'''