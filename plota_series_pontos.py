import xarray as xr
import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt

path = '/Volumes/BRENO_HD/GLOSS/'

def generic(plot= False):
    erros = []
    for ponto in os.listdir(path):
        serie = pd.read_csv(path+ ponto)
        try:
            serie.columns=['ano', 'mes', 'dia', 'hora', 'asm']

            d_index = []

            for i in range(len(serie)):
                ano = serie['ano'][i]
                mes = serie['mes'][i]
                dia = serie['dia'][i]
                hora = serie ['hora'][i]
                date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:00:00')
                d_index.append(date)

                if serie['asm'][i] < -5000 or serie['asm'][i] > 4000:
                    serie['asm'][i] = np.nan

            serie.index = d_index
            if plot:
                plt.figure()
                plt.plot(serie['asm'])
                plt.grid()
                plt.savefig('/Users/breno/Documents/Mestrado/estudos_dados/'+ponto[:-3] + '.png')
        except Exception as e:
            erros.append(ponto)
            print('ERRO EM ' + ponto)
            return erros

def generic2(erros, plot= False):
    erros2 = []
    for ponto in erros:
        try:
            serie = pd.read_csv(path+ ponto)
            d_index = []
            for i in range(len(serie)):
                ano = serie['ano'][i]
                mes = serie['mes'][i]
                dia = serie['dia'][i]
                hora = serie ['hora'][i]
                minu = serie['minuto'][i]
                date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:00')
                d_index.append(date)

                if serie['nivel'][i] < 0 or serie['nivel'][i] > 4000:
                    serie['nivel'][i] = np.nan

            serie.index = d_index

            if plot:
                plt.figure()
                plt.plot(serie['nivel'])
                plt.grid()
                plt.savefig('/Users/breno/Documents/Mestrado/estudos_dados/'+ponto[:-3] + '.png')
        except Exception:
            print('ERRO EM ' + ponto)
            erros2.append(ponto)
            return erros2

def cananeia(plot = False):
    ponto = 'Cananeia_gloss_brasil.csv'
    serie = pd.read_csv(path+ ponto)
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

        if serie['nivel'][i] < 1000 or serie['nivel'][i] > 4000:
            serie['nivel'][i] = np.nan

    serie.index = d_index

    if plot:
        plt.figure()
        plt.plot(serie['nivel'])
        plt.grid()
        plt.savefig('/Users/breno/Documents/Mestrado/estudos_dados/'+ponto[:-3] + '.png')

def export_serie(path, pontos_interesse = None):
    pontos_dado = {
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
    
    excecoes = []
    series = []
    for ponto in os.listdir(path):
        # print(ponto)
        # tem que usar um dicionario aqui pra associar a latlon dos pontos. o arquivo em si 
        # SUPOSTAMENTE (verificar) nao contem essa info
        try:
            serie = pd.read_csv(path+ ponto)
            serie.columns=['ano', 'mes', 'dia', 'hora', 'asm']

            d_index = []

            for i in range(len(serie)):
                ano = serie['ano'][i]
                mes = serie['mes'][i]
                dia = serie['dia'][i]
                hora = serie ['hora'][i]
                date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:00:00')
                d_index.append(date)
                for point in pontos_dado:
                    if point.lower() in ponto.lower():
                        lat = pontos_dado[point][0]
                        break
                        # lon = pontos_dado[point][1]
                        # serie['lat'] = lat
                        # serie['lon'] = lon

                # if serie['asm'][i] < -5000 or serie['asm'][i] > 4000:
                #     serie['asm'][i] = np.nan

            series.append((d_index[0], d_index[-1], ponto, lat))

        except Exception as e:
            excecoes.append(ponto)
    
    # generico 2

    excecoes_generico = []

    for ponto in excecoes:
        print(ponto)
        try:
            serie = pd.read_csv(path+ ponto)

            d_index = []

            for i in range(len(serie)):
                ano = serie['ano'][i]
                mes = serie['mes'][i]
                dia = serie['dia'][i]
                hora = serie ['hora'][i]
                minu = serie['minuto'][i]
                date = pd.to_datetime(f'{ano}-{mes}-{dia} {hora}:{minu}:00')
                d_index.append(date)
                for point in pontos_dado:
                    if point.lower() in ponto.lower():
                        lat = pontos_dado[point][0]
                        break
                        # lon = pontos_dado[point][1]
                        # serie['lat'] = lat
                        # serie['lon'] = lon

                # if serie['asm'][i] < -5000 or serie['asm'][i] > 4000:
                #     serie['asm'][i] = np.nan

            series.append((d_index[0], d_index[-1], ponto, lat))

        except Exception as e:
            # print('erro em ' + ponto)
            excecoes_generico.append(ponto)

    # cananeia

    for ponto in excecoes_generico:
    # ponto = 'Cananeia_gloss_brasil.csv'
        try:
            serie = pd.read_csv(path+ ponto, encoding='latin-1')
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

                for point in pontos_dado:
                    if point.lower() in ponto.lower():
                        lat = pontos_dado[point][0]
                        break


            series.append((d_index[0], d_index[-1], ponto, lat))
        except Exception as e:
            1+1

    return series
