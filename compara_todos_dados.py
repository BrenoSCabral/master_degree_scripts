import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os

# definindo funcao que plota o periodo

def plot_series(serie, figname='todas_series', xlim=False):
    '''
    Plota as séries temporais de uma lista de séries temporais.
    A serie tem que estar no formato [data_inicio, data_fim, lat]
    '''
    fig, ax = plt.subplots()
    # Plotar as linhas horizontais para cada item da lista
    for i in serie:
        ax.hlines(y=i[-1], xmin=i[0], xmax=i[1], color='blue')
        # ax.text(data_inicio, i, nome, ha='right', va='center')

    # Definir os rótulos dos eixos
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([pd.Timestamp("19450101"), pd.Timestamp("20250101")])
    ax.set_xlabel('Data')

    # Definir o título do gráfico
    ax.set_title('Período de Tempo todas as séries obtidas')

    # Rotacionar os rótulos do eixo x para melhorar a legibilidade
    plt.xticks(rotation=45)
    plt.grid()
    plt.ylabel('Latitude')

    # Exibir o gráfico
    plt.tight_layout()
    plt.savefig('/Users/breno/Documents/Mestrado/estudos_dados/' + figname + '.png')

# pegando os dados da marinha:
import plota_dados_marinha as marinha
path = '/Users/breno/Downloads/Dados'
serie_marinha = []
for i in os.listdir(path):
    df = pd.read_csv(path + '/' + i, skiprows=11, sep=';',encoding='latin-1')
    df.loc[len(df)] = df.columns.to_list()
    df.columns = ['data', 'ssh', 'Unnamed: 2']
    df = df.drop(columns=['Unnamed: 2'])

    # Converter 'data' para formato datetime, se necessário
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y %H:%M')
    df.set_index('data', inplace=True)

    df['ssh'] = df['ssh'].astype(int)

    df = df.sort_index()

    lat_lon = marinha.get_lat_lon(i)    
    serie_marinha.append((df.index[0], df.index[-1], i, lat_lon[0])) 

# pegando os dados do gloss:
import plota_series_pontos as gloss
path = '/Volumes/BRENO_HD/GLOSS/'
serie_gloss = gloss.export_serie(path)

plot_series(serie_marinha + serie_gloss)

# NNE atual
serie_total = serie_marinha + serie_gloss
nnea = []
for i in serie_total:
    if i[-1]>-20 and i[1] > pd.to_datetime('2000-01-01 00:00:00'):
        nnea.append(i)

plot_series(nnea, 'nne_dps_2000', [pd.Timestamp("20000101"), pd.Timestamp("20250101")])


#     import matplotlib.pyplot as plt
#     import matplotlib.dates as mdates
# def tick_ano_gpt(serie, figname='todas_series', xlim=False):

#     fig, ax = plt.subplots()

#     # Plotar as linhas horizontais para cada item da lista
#     for i in serie:
#         ax.hlines(y=i[-1], xmin=i[0], xmax=i[1], color='blue')
#         # ax.text(data_inicio, i, nome, ha='right', va='center')

#     # Definir os rótulos dos eixos
#     if xlim:
#         plt.xlim(xlim)
#     else:
#         plt.xlim([pd.Timestamp("19450101"), pd.Timestamp("20250101")])
#     ax.set_xlabel('Data')

#     # Definir o título do gráfico
#     ax.set_title('Período de Tempo todas as séries obtidas')

#     # Configurar o espaçamento dos ticks no eixo x para cada ano
#     years = mdates.YearLocator()
#     ax.xaxis.set_major_locator(years)

#     # Formatador para exibir os anos como 'YYYY'
#     year_format = mdates.DateFormatter('%Y')
#     ax.xaxis.set_major_formatter(year_format)

#     # Rotacionar os rótulos do eixo x para melhorar a legibilidade
#     plt.xticks(rotation=45)
#     plt.grid()
#     plt.ylabel('Latitude')

#     # Exibir o gráfico
#     plt.tight_layout()
#     plt.savefig('/Users/breno/Documents/Mestrado/estudos_dados/' + figname + '.png')
