
from read_data import all_series, sep_series
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import mdates

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


def plot_used_data(series):

    # for serie in sel_series:
    #     if series[serie]['lat'][0] < -4:
    #         continue
    #     elif serie[0] == 'F' or serie == 'PORTO DE MUCURIPEm0':
    #         continue
    #     else:
    #         del series[serie]
            

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    import numpy as np

    coord = ccrs.PlateCarree()

    # Criar figura com 2 subplots lado a lado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                gridspec_kw={'width_ratios': [1, 2]},
                                subplot_kw={'projection': coord})

    # =====================================
    # Primeiro plot: Mapa
    # =====================================
    ax1.add_feature(cfeature.LAND)
    ax1.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax1.add_feature(cfeature.BORDERS, zorder=10)
    ax1.add_feature(cfeature.COASTLINE, zorder=10)
    ax1.set_extent([-55, -30, -40, 5], crs=coord)  # Ajustado para lat negativo correto

    # Plotar pontos
    for point in series:
        pt = series[point]
        ax1.plot(pt['lon'][0], pt['lat'][0], color='red', marker='o',
                linewidth=2, transform=coord)

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

    # Sincronizar limites de latitude
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_yticklabels([])
    # Formatar datas no eixo X
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Ajustar layout
    plt.subplots_adjust(wspace=0.03)
    plt.tight_layout()
    plt.savefig('/Users/breno/mestrado/available_points.png', dpi=300)