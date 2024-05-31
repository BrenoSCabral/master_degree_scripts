import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os


def passa_baixa(modelo, plot, fig_folder):
    # vai ser aplicado em dois passos pra mim: primeior passa baixa c freq horaria
    # e depois passa alta c freq diaria
    if modelo == False:
        dt = 1/24  # Frequência de amostragem (1 hora)
        pesos = 10
    else:
        dt = 1
        pesos = 44
    periodo_min = 3  # Período mínimo em dias
    cutoff_max = 1 / periodo_min
    fn = 1/(2*dt) # frequencia de Nyquist

    b, a = signal.butter(pesos, cutoff_max/fn,
                        btype='low')  # Filtro passa-banda de ordem 2


    if plot:
        plot_filter(b, a, fig_folder)

    return (b,a)


def passa_banda(modelo, plot, fig_folder):
    # Filtragem passa-banda entre 3 e 30 dias
    if modelo == False:
        dt = 1/24  # Frequência de amostragem (1 hora)
        pesos = 4
    else:
        dt = 1
        pesos = 15
    periodo_min = 3  # Período mínimo em dias
    periodo_max = 30  # Período máximo em dias
    cutoff_max = 1 / periodo_min
    cutoff_min = 1 / periodo_max
    fn = 1/(2*dt) # frequencia de Nyquist
    b, a = signal.butter(pesos, [cutoff_min/fn, cutoff_max/fn],
                        btype='bandpass')  # Filtro passa-banda de ordem 2

    if plot:
        plot_filter(b, a, fig_folder)

    return (b,a)


def passa_alta(plot, fig_folder):
    '''
    Nesse caso, precisa rodar para o ssh já reamostrado em 1 dia
    '''
    dt = 1  # Frequência de amostragem (1 hora)
    periodo = 30  # Período mínimo em dias
    cutoff = 1 / periodo
    fn = 1/(2*dt) # frequencia de Nyquist

    b, a = signal.butter(14, cutoff/fn,
                        btype='high')  # Filtro passa-banda de ordem 2
    if plot:
        plot_filter(b, a, fig_folder)

    return (b,a)


def aplica_filtro(b,a, nivel_mar, plot=False, tempo=None, nome_serie=None, fig_folder=None):
    nivel_mar_filtrado = signal.filtfilt(b, a, nivel_mar)
    if plot:
        plot_series(tempo, nivel_mar, nivel_mar_filtrado, nome_serie, fig_folder)

    return nivel_mar_filtrado


def filtra_dados(nivel_mar, tempo,  metodo, modelo = False, plot=False, fig_folder=None, nome_serie=None):
    if metodo ==  'band':
        b, a = passa_banda(modelo, plot, fig_folder)
    elif metodo == 'low':
        b, a = passa_baixa(modelo, plot, fig_folder)
    elif metodo == 'high':
        b, a = passa_alta(plot, fig_folder)

    nivel_mar_filtrado = aplica_filtro(b,a, nivel_mar, tempo, nome_serie, plot, fig_folder)
    return nivel_mar_filtrado


def plot_filter(b, a, fig_folder):
    frequencias, resposta = signal.freqz(b, a)
    plt.figure()
    plt.semilogx(1/(frequencias*fn/np.pi), np.abs(resposta))
    plt.title('Resposta de Frequência do Filtro Digital Passa Baixa')
    plt.xlabel('Frequência Normalizada')
    plt.ylabel('Magnitude')
    plt.grid(which='both', axis='both')
    plt.savefig(f'{fig_folder}filtro_passa_baixa.png')


def plot_series(tempo, nivel_mar, nivel_mar_filtrado, nome_serie, fig_folder):
    plt.ion()
    # Plot os sinais originais e filtrados
    plt.figure(figsize=(10, 6))
    plt.plot(tempo, nivel_mar, label='Sinal original')
    plt.plot(tempo, nivel_mar_filtrado, label='Sinal filtrado')
    plt.xlabel('Tempo')
    plt.ylabel('Nível do mar')
    plt.title(f'Filtragem de sinal de nível do mar - {nome_serie}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{fig_folder}{nome_serie}_filtrado.png')

    # Plot os sinais originais e filtrados sobrepostos

    plt.figure(figsize=(10, 6))
    plt.plot(tempo, nivel_mar - nivel_mar.mean(), label='Sinal original - média')
    plt.plot(tempo, nivel_mar_filtrado, label='Sinal filtrado')
    plt.xlabel('Tempo')
    plt.ylabel('Nível do mar')
    plt.title(f'Filtragem de sinal de nível do mar - {nome_serie}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{fig_folder}{nome_serie}_filtrado_sobreposto.png')

    plt.close('all')
