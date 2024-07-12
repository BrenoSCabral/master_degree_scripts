from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq
import os
import numpy as np
import sys

sys.path.append(
    'old'
)
from crosspecs import crospecs


# plot de serie temporal
def plot_time_series(serie, title, path, figname):
    plt.figure(figsize=(12,8))
    plt.plot(serie)
    plt.title(title)

    plt.grid()


    os.makedirs(path, exist_ok=True)
    plt.savefig(path + figname)


# plot de espectro
def plot_spectrum(seri1, title, path, figname, is_data=True):
    
    seri1 = np.asarray(seri1)
    fig = plt.figure(figsize=(14,8))
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    N1 = len(seri1) # numero de medicoes
    if is_data:
        T1 = 1.0/24 # frequencia das medicoes (nesse caso = 1 medicao por h)
    else:
        T1 = 1.0
    yif1 = fft(seri1)
    xif1 = fftfreq(N1,T1)
    tif1 = 1/xif1

    ax.semilogx(tif1, 2.0/N1 * np.abs(yif1), label = 'Original')

    ax.legend()

    ax.grid()

    # ax.set_xticklabels([3, 5, 10, 20, 30, 40])

    ax.set_ylabel('Densidade Espectral [cm²/dia]')
    ax.set_xlabel('Dias')
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + figname)


def plot_double_spectrum(data, filtered, title, path, figname, is_data = True):
    data = np.asarray(data)


    fig = plt.figure(figsize=(14,8))
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    N1 = len(data) # numero de medicoes
    if is_data:
        T1 = 1.0/24 # frequencia das medicoes (nesse caso = 1 medicao por h)
    else:
        T1 = 1.0
    yif1 = fft(data)
    xif1 = fftfreq(N1,T1)
    tif1 = 1/xif1


    N = len(filtered) # numero de medicoes
    T = 1.0 # frequencia das medicoes (nesse caso = 1 medicao a cada 24h)
    yif = fft(filtered)
    xif = fftfreq(N,T)
    tif = 1/xif

    ax.semilogx(tif1, 2.0/N1 * np.abs(yif1), label = 'Original')
    ax.semilogx(tif, 2.0/N * np.abs(yif), label = 'Filtrado')

    # ax.set_xlim(10,20)
    # ax.set_xticks([3, 5, 10, 20, 30, 40]) 

    # ax.set_xticklabels([])
    ax.legend()
    # ax.set_yticks([])

    # ax.ylim([-0.1, 40])
    ax.grid()

    # ax.set_xticklabels([3, 5, 10, 20, 30, 40])

    ax.set_ylabel('Densidade Espectral [cm²/dia]')
    ax.set_xlabel('Dias')
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + figname)


# plot de duas series -> modelo vs dado
def compare_time_series(data, model, title, path, figname):
    plt.figure(figsize=(12,8))
    plt.plot(data, label = 'Dado')
    plt.plot(model, label = 'Modelo')
    plt.grid()
    plt.title(title)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + figname)


# comparacao de dois espectros -> aquele que o PP me passou
def compare_spectra(serie1, serie2, title, path, figname):
    xx1=serie1
    xx2=serie2
    ppp=len(xx1)
    dt=24#diario
    win=2
    smo=999
    ci=99
    h1,h2,fff,coef,conf,fase=crospecs(xx1, xx2, ppp, dt, win, smo, ci)

    # plt.figure(1)
    fig = plt.figure(figsize=(8,12))
    plt.plot(1./fff/24,coef,'b')
    plt.plot(1./fff/24,conf,'--k')
    plt.xlim([0,40])
    plt.grid()
    plt.title(title)
    os.makedirs(path, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path + figname)

