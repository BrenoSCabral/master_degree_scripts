import sys
sys.path.append('../')
from read_data import all_series, sep_series
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from matplotlib.lines import Line2D


def get_all_available_data(path_data = '/home/bcabral/mestrado/data'):
    treated = all_series(path_data)
    series = {}
    checked_series = {}
    for serie in treated:
        # if treated[serie]['lat'][0] > -20:
        #     continue
        if treated[serie].index[-1] < pd.Timestamp("1995"):
            continue
        if treated[serie].index[0] > pd.Timestamp("2020"):
            continue
        if treated[serie].index[-1] > pd.Timestamp("2020"):
            if len(treated[serie][:'2019-12-31']) < 24*30*6:
                continue
            else:
                treated[serie] = treated[serie][:'2019-12-31']
        series[serie] = treated[serie]
    
    sep_serie = sep_series(series)

    # check depois de ter os nans
    for serie in sep_serie:
        # if sep_serie[serie]['lat'][0] > -20:
        #     continue
        if sep_serie[serie].index[-1] < pd.Timestamp("1995"):
            continue
        if sep_serie[serie].index[0] < pd.Timestamp("1995"):
            if len(sep_serie[serie]['1995-01-01':]) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie]['1995-01-01':]
        if sep_serie[serie].index[0] > pd.Timestamp("2020"):
            continue
        if sep_serie[serie].index[-1] > pd.Timestamp("2020"):
            if len(sep_serie[serie][:'2019-12-31']) < 24*30*6:
                continue
            else:
                sep_serie[serie] = sep_serie[serie][:'2019-12-31']

        checked_series[serie] = sep_serie[serie]

    sel_series = {k: v for k, v in checked_series.items() if len(v) >= 24*30*6}

    return sel_series


def acro_full(name):
    if name == 'TIPLAMm0':
        return 'TIPLAM'
    elif name == 'Imbituba_2001_20074':
        return 'Imbituba'
    elif name == 'ubatuba22':
        return 'Ubatuba'
    elif name == 'TEPORTIm0':
        return 'TEPORTI'
    elif name == 'BARRA DE PARANAGUÁ - CANAL DA GALHETAm0':
        return 'B. Paranaguá'
    elif name == 'PORTO DE PARANAGUÁ - CAIS OESTEm0':
        return 'Paranaguá Port'
    elif name == 'PORTO DE MUCURIPEm0':
        return "Mucuripe's Port"
    elif name == 'ilha_fiscal12':
        return "Ilha Fiscal"
    elif name == 'Salvador_2004_20150':
        return 'Salvador'
    elif name == 'PORTO DO FORNOm0':
        return "Forno's Port"
    elif name == 'Macae_2001_20073':
        return 'Macaé'
    elif name == 'rio_grande16':
        return 'Rio Grande'
    elif name == 'NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0':
        return "Del. of Itajaí's Ports"
    else:
        return ''
    

def acro(name):
    return ''
    if name == 'TERMINAL PORTUÁRIO DA PONTA DO FÉLIXm0':
        return 'TPF'
    if name == 'TIPLAMm0':
        return 'TIP'
    elif name == 'Imbituba_2001_200719':
        return 'IMB'
    elif name == 'ubatuba34':
        return 'UBA'
    elif name == 'TEPORTIm0':
        return 'TEP'
    elif name == 'BARRA DE PARANAGUÁ - CANAL DA GALHETAm0':
        return 'BPA'
    elif name == 'PORTO DE PARANAGUÁ - CAIS OESTEm0':
        return 'PAR'
    elif name == 'PORTO DE MUCURIPEm0':
        return "MUP"
    elif name == 'ilha_fiscal12':
        return "IFI"
    elif name == 'Salvador_2004_20150':
        return 'SAL'
    elif name == 'PORTO DO FORNOm0':
        return "FOP"
    elif name == 'Macae_2001_20073':
        return 'MAC'
    elif name == 'rio_grande16':
        return 'RIG'
    elif name == 'NOVA DEL DA CAP DOS PORTOS EM ITAJAÍ m0':
        return "DIP"
    else:
        return ''


def set_txt_coords(pt, name):
    rot = 0
    if name == "MUP":
        pt_lon = pt['lon'][0] - 2
        pt_lat = pt['lat'][0] - 1
    elif name == 'SAL':
        pt_lat = pt['lat'][0] + 0.2
        pt_lon = pt['lon'][0] - 2
    elif name == 'MAC':
        pt_lat = pt['lat'][0] 
        pt_lon = pt['lon'][0] + 0.7
    elif name == "FOP":
        pt_lat = pt['lat'][0] - .7
        pt_lon = pt['lon'][0] + .5
        rot = 0
    elif name == 'IFI':
        pt_lat = pt['lat'][0] -.7
        pt_lon = pt['lon'][0] -.2
        rot = -45
    elif name == 'UBA':
        pt_lat = pt['lat'][0] - 0.5
        pt_lon = pt['lon'][0] + .1
        rot = -45
    elif name =='TIP':
        pt_lat = pt['lat'][0] - 1
        pt_lon = pt['lon'][0] -.1
        rot = -40
    elif name== 'PAR':
        pt_lat = pt['lat'][0] +.5
        pt_lon = pt['lon'][0] -1
    elif name == 'TPF':
        pt_lat = pt['lat'][0] - .3
        pt_lon = pt['lon'][0] -1.5
    elif name == 'BPA':
        pt_lat = pt['lat'][0] - .5
        pt_lon = pt['lon'][0] +.4
    elif name =='TEP':
        pt_lat = pt['lat'][0] -.5
        pt_lon = pt['lon'][0] -1.7
    elif name == "DIP":
        pt_lat = pt['lat'][0] -.7
        pt_lon = pt['lon'][0] +.7
    elif name =='RIG':
        pt_lat = pt['lat'][0] -.7
        pt_lon = pt['lon'][0] + .5
    elif name =='IMB':
        pt_lat = pt['lat'][0] - 1.3
        pt_lon = pt['lon'][0] + 0.2

    else:
        print(name)
        pt_lat = pt['lat'][0] - .5
        pt_lon = pt['lon'][0] + 0.2


    return (pt_lon, pt_lat, rot)



def set_txt_coords_full(pt, name):
    rot = 0
    if name == "Mucuripe's Port":
        pt_lon = pt['lon'][0] - 7
        pt_lat = pt['lat'][0] - 1
    elif name == 'Salvador':
        pt_lat = pt['lat'][0] + 0.2
        pt_lon = pt['lon'][0] - 3.8
    elif name == 'Macaé':
        pt_lat = pt['lat'][0] + 0.2
        pt_lon = pt['lon'][0] + 0.5
    elif name == "Forno's Port":
        pt_lat = pt['lat'][0] - .5
        pt_lon = pt['lon'][0] + .4
        rot = 0
    elif name == 'Ilha Fiscal':
        pt_lat = pt['lat'][0] -.5
        pt_lon = pt['lon'][0] -.2
        rot = -45
    elif name == 'Ubatuba':
        pt_lat = pt['lat'][0] - 0.2
        pt_lon = pt['lon'][0] + .1
        rot = -45
    elif name =='TIPLAM':
        pt_lat = pt['lat'][0] - 1
        pt_lon = pt['lon'][0] + .1
        rot = -40
    elif name== 'Paranaguá Port':
        pt_lat = pt['lat'][0] +.5
        pt_lon = pt['lon'][0] - 5.5
    elif name == 'B. Paranaguá':
        pt_lat = pt['lat'][0] - .5
        pt_lon = pt['lon'][0] - 5.5
    elif name =='TEPORTI':
        pt_lat = pt['lat'][0] -.5
        pt_lon = pt['lon'][0] -4
    elif name == "Del. of Itajaí's Ports":
        pt_lat = pt['lat'][0] -.7
        pt_lon = pt['lon'][0] +.2
    elif name =='Rio Grande':
        pt_lat = pt['lat'][0] 
        pt_lon = pt['lon'][0] + 1.4
    elif name =='Imbituba':
        pt_lat = pt['lat'][0] - 1.3
        pt_lon = pt['lon'][0] + 0.2

    else:
        print(name)
        pt_lat = pt['lat'][0] - .5
        pt_lon = pt['lon'][0] + 0.2


    return (pt_lon, pt_lat, rot)


def plot_used_data(series):

    # for serie in sel_series:
    #     if series[serie]['lat'][0] < -4:
    #         continue
    #     elif serie[0] == 'F' or serie == 'PORTO DE MUCURIPEm0':
    #         continue
    #     else:
    #         del series[serie]


    coord = ccrs.PlateCarree()

    # Criar figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                gridspec_kw={'width_ratios': [1, 2], 'wspace': 0.0},
                                subplot_kw={'projection': coord})

    # =====================================
    # Primeiro plot: Mapa
    # =====================================
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax1.add_feature(cfeature.BORDERS, zorder=10)
    ax1.add_feature(cfeature.COASTLINE, zorder=10)
    ax1.set_extent([-54, -34, -35, 5], crs=coord)  # Ajustado para lat negativo correto

    # Plotar pontos
    # for point in series:
    #     pt = series[point]
    #     ax1.plot(pt['lon'][0], pt['lat'][0], color='red', marker='o',
    #             linewidth=2, transform=coord)

    texts = []
    for point in series:
        pt = series[point]
        ax1.plot(pt['lon'][0], pt['lat'][0], color='red', marker='o',
                linewidth=2, transform=coord)
        txtcoords = set_txt_coords(pt,acro(point))
        text = ax1.text(txtcoords[0], txtcoords[1],  acro(point), 
                        transform=coord, ha='left', va='bottom', fontsize=8, rotation = txtcoords[2], rotation_mode='anchor')
        texts.append(text)


    # Configurar grade do mapa
    lons = np.arange(-70, -20, 5)
    lats = np.arange(-35, 10, 5)
    gl = ax1.gridlines(crs=coord, linewidth=.5, color='black', alpha=0.5,
                    linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = True
    gl.xlocator = mticker.FixedLocator(lons)
    gl.ylocator = mticker.FixedLocator(lats)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    # =====================================
    # Segundo plot: Séries temporais
    # =====================================
    # Remover projeção geográfica para o plot temporal
    ax2 = fig.add_subplot(122, sharey=ax1)  # Compartilhar eixo Y (latitudes)

    # Plotar séries temporais
    for serie in series:
        ax2.plot(series[serie].index, series[serie]['lat'], color='red')

    # Configurar eixos
    # ax2.set_xlabel('Data')
    ax2.grid(True, linestyle='--', alpha=0.5)


    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_locator(mticker.FixedLocator(np.arange(-35, 5, 5)))  # Same ticks as map
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(LATITUDE_FORMATTER))  # Same formatter
    ax2.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)

    # ax2.set_yticklabels([])
    # Formatar datas no eixo X
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Ajustar layout
    # plt.subplots_adjust(wspace=0.03)
    plt.tight_layout()
    # plt.show()
    plt.savefig('/Users/breno/mestrado/available_points.png', dpi=300)


def plota_pontos(pts):
    import os
    from read_data import read_gloss
    from read_data import nome_estacao
    from read_data import read_marinha
    from read_data import read_simcosta

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

    pts = infos
    coord = ccrs.PlateCarree()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection=coord)


    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.BORDERS, zorder=10)
    ax.add_feature(cfeature.COASTLINE, zorder=10)
    ax.set_extent([-60, -30, 5, -40], crs=ccrs.PlateCarree())


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
               markersize=12, linestyle='None'),
        Line2D([0],[0], color = 'lime', label='Navy', marker ='P', markerfacecolor='lime',
               markersize=12, linestyle='None'),
        Line2D([0],[0], color = 'yellow', label='GOOS', marker ='P', markerfacecolor='yellow',
               markersize=12, linestyle='None'),
        Line2D([0],[0], color = 'r', label='GLOSS', marker ='o', markerfacecolor='r',
               markersize=12, linestyle='None')   ,
        # Line2D([0],[0], color = 'darkorange', label='Utilizado', marker ='s', markerfacecolor='darkorange',
        #        markersize=12, linestyle='None')              
    ]

    ax.legend(handles = legend_elements, loc='lower right')

    plt.tight_layout()
    # plt.show()
    plt.savefig('/Users/breno/mestrado/all_points.png')
    # plt.savefig(f'/Users/breno/Documents/Mestrado/dados/estudo/pontos_{namefile}.png')



series = get_all_available_data('/Users/breno/Documents/Mestrado/resultados/data')

series.pop('Fortaleza22')
series.pop('Fortaleza1')
series.pop('Fortaleza3')
series.pop('CIA DOCAS DO PORTO DE SANTANAm0')
series.pop('PAGA DÍVIDAS Im0')
series.pop('PORTO DE VILA DO CONDEm0')
series.pop('CANIVETEm0')
series.pop('PONTA DO PECEMm0')
series.pop('IGARAPÉ GRD DO CURUÁm0')
series.pop('SERRARIA RIO MATAPIm0')
series.pop('salvador20')
series.pop('salvador21')
series.pop('salvador22')
series.pop('salvador23')
series.pop('salvador25')
series.pop('salvador83')
series.pop('salvador85')
series.pop('salvador213')
series.pop('salvador214')
series.pop('salvador215')
series.pop('salvador216')
series.pop('salvador217')
series.pop('Salvador_glossbrasil25')
series.pop('Salvador_glossbrasil27')
series.pop('Salvador_glossbrasil83')
series.pop('Salvador_glossbrasil85')
series.pop('CAPITANIA DE SALVADORm0')





# for i in series.keys():
#     if series[i]['lat'][0] >-4 and i != 'PORTO DE MUCURIPEm0':
#         #series.pop(i)
#         print(i)


plot_used_data(series)