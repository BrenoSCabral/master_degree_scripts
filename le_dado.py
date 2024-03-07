import xarray as xr
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
import time
from scipy.fft import fft, fftfreq
path.append(
    '/Users/breno/Documents/Mestrado/tese/scripts'
)
from filtro import filtra_dados


# le dados nc:
def le_dado_nc(filename, path_dado):
    '''
        filename: (string) nome do arquivo csv

        retorna um Dataset com os dados
    '''
    return (xr.open_dataset(path_dado + filename))

def le_dado_csv(filename, path_dado, separador=';', pula_linhas = None):
    '''
        OBS: esse pode mudar de acordo com a estrutura do csv
        filename: (string) nome do arquivo csv

        retorna um Dataset com os dados
    '''
    df_file = pd.read_csv(path_dado + filename,
                          sep=separador, header=None, skiprows=pula_linhas)
    df_file.columns = ['time', 'sea_level','drop']
    df_file.index = pd.to_datetime(df_file['time'], format="%d/%m/%Y %H:%M")
    df_file = df_file.drop('drop', axis=1)
    df_file = df_file.drop('time', axis=1)

    return(xr.Dataset.from_dataframe(df_file))

def corta_dado(dataini, datafim, dado):
    variaveis = list(dado.data_vars) # VER NO FUTURO SE ISSO PERMANECE!
    variavel = variaveis [0]
    # while variavel not in variaveis:
    #     print('-', '\n- '.join(variaveis))
    #     variavel = input('\nQual das variáveis acima você deseja acessar? ')
    #     if variavel not in variaveis:
    #         print("\n",
    #               "Opção inválida. Por favor, escolha uma das variáveis listadas.",
    #               "\n")
    #         time.sleep(1)
    dado_cortado = dado[variavel].sel(time=slice(dataini, datafim))
    datas_cortadas = dado.time.sel(time=slice(dataini, datafim))

    return(dado_cortado, datas_cortadas)

def reamostra_dado_old(dado, dado_tempo):
    '''
        VELHO!
        Afonso recomendou pegar o valor singular, ao invés de interpolar cubicamente
        Pega um array numpy de dado e outro de tempo e reamostra para 1 dia.
        dado: (numpy array) array com dados (de ssh)
        dado_tempo: (numpy array) array de datas
    '''

    da = xr.DataArray(data=dado,
                      dims=['time'],
                      coords=dict(time=dado_tempo))
    '''
    metodos de interpolacao:
    https://xarray.pydata.org/en/v0.14.1/generated/xarray.DataArray.resample.html
    '''
    da_diario = da.resample(time='1D').interpolate('cubic')
    return da_diario

def reamostra_dado(dado, dado_tempo):
    '''
        Reamostra o dado pegando o valor das 00
        diferente do anterior que fazia uma interpolacao cubica pra
        gerar um resultado diario

    Args:
        dado (numpy array): Serie temporal de asm
        dado_tempo (Data Array): Serie de tempo

    Returns:
        numpy array: Serie temporal reamostrada com os horarios das 00
    '''

    tempo = dado_tempo.values
    datas = pd.date_range(tempo[0], tempo[-1], freq = 'h')
    df_dates = pd.DataFrame({'data':dado}, index = datas)
    return np.asarray(df_dates[df_dates.index.hour==00]['data'])

def pega_fft(ssh, nome_serie, fig_folder):
    N = 365*24 # numero de medicoes
    T = 1.0 / 24 # frequencia das medicoes (nesse caso = 1 medicao a cada 24h)
    yf = fft(ssh) # faz a transformada de fourier
    xf = fftfreq(N, T) # freequencia de amostragem da transformada
    # por algum motivo, o exemplo usa esse xfa, mas eu nao achei que valeria a pena
    # xfa = xf[:N//2]
    # tfa = 1/xfa
    tf = 1/xf

    # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    # ax1.semilogx(tfa, 2.0/N * np.abs(yf[0:N//2]))
    # ax1.set_ylim([-0.1, 40])
    # ax1.grid()
    fig = plt.figure(figsize=(20, 7))
    plt.semilogx(tf, 2.0/N * np.abs(yf))
    plt.ylim([-0.1, 40])
    plt.ylabel('Energia')
    plt.xlabel('Dias')
    plt.title('Decomposição das frequências da série de '+nome_serie)
    plt.grid()
    plt.savefig(f'{fig_folder}{nome_serie}_frequencias.png')

def pega_fft_diario(ssh, nome_serie, fig_folder):
    N = 365 # numero de medicoes
    T = 1.0 # frequencia das medicoes (nesse caso = 1 medicao a cada 24h)
    yf = fft(ssh) # faz a transformada de fourier
    xf = fftfreq(N, T) # freequencia de amostragem da transformada
    # por algum motivo, o exemplo usa esse xfa, mas eu nao achei que valeria a pena
    # xfa = xf[:N//2]
    # tfa = 1/xfa
    tf = 1/xf

    # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    # ax1.semilogx(tfa, 2.0/N * np.abs(yf[0:N//2]))
    # ax1.set_ylim([-0.1, 40])
    # ax1.grid()
    fig = plt.figure(figsize=(20, 7))
    plt.semilogx(tf, 2.0/N * np.abs(yf))
    plt.ylim([-0.1, 40])
    plt.ylabel('Energia')
    plt.xlabel('Dias')
    plt.title('Decomposição das frequências da série de '+nome_serie)
    plt.grid()
    plt.savefig(f'{fig_folder}{nome_serie}_frequencias.png')

def roda_analise(filename, path_dado, formato, nome, fig_folder,
                 dataini = None, datafim = None):
    formatos = ['csv', 'nc']
    while formato not in formatos:
        formato = input('Vc deseja abrir um csv ou um nc? ')
        formato = formato.lower()
        if formato not in formatos:
            print('\nfavor escolher um formato válido.\n')
            time.sleep(1)
    if formato == 'csv':
        dado = le_dado_csv(filename, path_dado)
    elif formato == 'nc':
        dado = le_dado_nc(filename, path_dado)
    
    if dataini == None:
        dataini = dado.time[0].values
    if datafim == None:
        datafim = dado.time[-1].values

    dado_ssh, dado_tempo = corta_dado(dataini, datafim, dado)

    # metodo = 'composto' # gambiarra feita no desespero
    # if metodo == 'composto':
    #     dado_filtrado = filtra_dados(dado_ssh, dado_tempo, nome, fig_folder, 'low')
    #     filtro_reamostrado = reamostra_dado(dado_filtrado, dado_tempo)
    #     dado_filtrado2 = filtra_dados(filtro_reamostrado.data, filtro_reamostrado['time'],
    #                                   nome, fig_folder, 'high')


    return dado_ssh, dado_tempo

