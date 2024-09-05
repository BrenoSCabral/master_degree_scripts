import pandas as pd
from matplotlib import pyplot as plt
import os
import matplotlib.dates as mdates
import numpy as np

def prepare_hovmoller_data(df_ssh):
    # testar essa linha abaixo
    df_pivot = df_ssh.pivot_table(index='lat', columns='time', values='ssh', aggfunc=np.mean)
    
    # Ordenar pelas latitudes e tempos
    df_pivot = df_pivot.sort_index(ascending=True)
    df_pivot = df_pivot.sort_index(axis=1, ascending=True)
    
    return df_pivot


def plot_hovmoller(hovmoller_data, model, fig_folder):
    # otimizado pra plotar um ano INTEIRO de jan a dez
    ano = hovmoller_data.columns[0].year
    mid = datetime.datetime(ano,7,1)


    data = [
    hovmoller_data.loc[:, (hovmoller_data.columns < mid)],
    hovmoller_data.loc[:, (hovmoller_data.columns >= mid)]
    ]

    fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8), constrained_layout=True)
    levels = np.linspace(-20, 20, 100)

    for ax, hov_data in zip(axes, data):
        
        hov_data= np.clip(hov_data, -20, 20)

        # Define os níveis de SSH para o contorno contínuo
        
        # Plotar o preenchimento de contorno contínuo
        c = ax.contourf(hov_data.columns, hov_data.index, hov_data.values, levels=levels, cmap='bwr')

        
        # Adicionar contornos preto para valores extremos
        ax.contour(hov_data.columns, hov_data.index, hov_data.values, levels=[-40,-30,-20,-10, 10,20,30,40], colors='black', linestyles='-', linewidths=.5)
        

        ax.set_ylabel('Latitude')



        # Configurar o formato do eixo X para exibir os meses abreviados
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' para meses abreviados (Jan, Feb, Mar...)

        ticks = pd.date_range(start=hov_data.columns[0], end=hov_data.columns[-1], freq='MS')
        ax.set_xticks(ticks)

        ax.tick_params(axis='x', rotation=0, labelsize=14)  # Alinhar os nomes dos meses sem rotação
        ax.tick_params(axis='y', labelsize=14)  # Aumentar o tamanho da fonte das latitudes



    # fig.suptitle('Diagrama de Hovmöller com Contornos Contínuos')

    # Adicionar uma colorbar grande ao lado dos subplots
    cbar = fig.colorbar(c, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.03)
    cbar.set_label('SSH (cm)', fontsize=14)
    cbar.set_ticks([-20, -10, 0, 10, 20])  # Definir ticks específicos
    cbar.ax.tick_params(labelsize=12)

    # plt.show()
    os.makedirs(f'/{fig_folder}', exist_ok=True)
    plt.savefig(f'/{fig_folder}/{model}_{ano}.png')
    # os.makedirs('/Users/breno/mestrado/hovmoller/', exist_ok=True)
    # plt.savefig(f'/Users/breno/mestrado/hovmoller/{model}_{ano}.png')


def plot_hovmoller_u20(hovmoller_data, model, fig_folder):
    # otimizado pra plotar um ano INTEIRO de jan a dez
    ano = hovmoller_data.columns[0].year
    mid = datetime.datetime(ano,7,1)


    data = [
    hovmoller_data.loc[:, (hovmoller_data.columns < mid)],
    hovmoller_data.loc[:, (hovmoller_data.columns >= mid)]
    ]

    fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8), constrained_layout=True)
    levels = np.linspace(-20, 20, 100)

    for ax, hov_data in zip(axes, data):
        
        hov_data= np.clip(hov_data, -20, 20)

        # Define os níveis de SSH para o contorno contínuo
        
        # Plotar o preenchimento de contorno contínuo
        c = ax.contourf(hov_data.columns, hov_data.index, hov_data.values, levels=levels, cmap='bwr')

        
        # Adicionar contornos preto para valores extremos
        ax.contour(hov_data.columns, hov_data.index, hov_data.values, levels=[-40,-30,-20,-10, 10,20,30,40], colors='black', linestyles='-', linewidths=.5)
        

        ax.set_ylabel('Latitude')



        # Configurar o formato do eixo X para exibir os meses abreviados
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' para meses abreviados (Jan, Feb, Mar...)

        ticks = pd.date_range(start=hov_data.columns[0], end=hov_data.columns[-1], freq='MS')
        ax.set_xticks(ticks)

        ax.tick_params(axis='x', rotation=0, labelsize=14)  # Alinhar os nomes dos meses sem rotação
        ax.tick_params(axis='y', labelsize=14)  # Aumentar o tamanho da fonte das latitudes



    # fig.suptitle('Diagrama de Hovmöller com Contornos Contínuos')

    # Adicionar uma colorbar grande ao lado dos subplots
    cbar = fig.colorbar(c, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.03)
    cbar.set_label('SSH (cm)', fontsize=14)
    cbar.set_ticks([-20, -10, 0, 10, 20])  # Definir ticks específicos
    cbar.ax.tick_params(labelsize=12)

    # plt.show()
    os.makedirs(f'/{fig_folder}', exist_ok=True)
    plt.savefig(f'/{fig_folder}/{model}_{ano}_sub20.png')
    # os.makedirs('/Users/breno/mestrado/hovmoller/', exist_ok=True)
    # plt.savefig(f'/Users/breno/mestrado/hovmoller/{model}_{ano}_sub20.png')


def plot_hovmoller_o20(hovmoller_data, model, fig_folder):
    # otimizado pra plotar um ano INTEIRO de jan a dez
    ano = hovmoller_data.columns[0].year
    mid = datetime.datetime(ano,7,1)


    data = [
    hovmoller_data.loc[:, (hovmoller_data.columns < mid)],
    hovmoller_data.loc[:, (hovmoller_data.columns >= mid)]
    ]

    fig,axes = plt.subplots(nrows=2, ncols=1, figsize=(14,8), constrained_layout=True)
    levels = np.linspace(-10, 10, 100)

    for ax, hov_data in zip(axes, data):
        
        hov_data= np.clip(hov_data, -10, 10)

        # Define os níveis de SSH para o contorno contínuo
        
        # Plotar o preenchimento de contorno contínuo
        c = ax.contourf(hov_data.columns, hov_data.index, hov_data.values, levels=levels, cmap='bwr')

        
        # Adicionar contornos preto para valores extremos
        ax.contour(hov_data.columns, hov_data.index, hov_data.values, levels=[-10, -5, 5- 10], colors='black', linestyles='-', linewidths=.5)
        

        ax.set_ylabel('Latitude')



        # Configurar o formato do eixo X para exibir os meses abreviados
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # '%b' para meses abreviados (Jan, Feb, Mar...)

        ticks = pd.date_range(start=hov_data.columns[0], end=hov_data.columns[-1], freq='MS')
        ax.set_xticks(ticks)

        ax.tick_params(axis='x', rotation=0, labelsize=14)  # Alinhar os nomes dos meses sem rotação
        ax.tick_params(axis='y', labelsize=14)  # Aumentar o tamanho da fonte das latitudes



    # fig.suptitle('Diagrama de Hovmöller com Contornos Contínuos')

    # Adicionar uma colorbar grande ao lado dos subplots
    cbar = fig.colorbar(c, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.03)
    cbar.set_label('SSH (cm)', fontsize=14)
    cbar.set_ticks([-10, 0, 10])  # Definir ticks específicos
    cbar.ax.tick_params(labelsize=12)

    # plt.show()
    os.makedirs(f'/{fig_folder}', exist_ok=True)
    plt.savefig(f'/{fig_folder}/{model}_{ano}_+20.png')
    # os.makedirs('/Users/breno/mestrado/hovmoller/', exist_ok=True)
    # plt.savefig(f'/Users/breno/mestrado/hovmoller/{model}_{ano}_+20.png')
