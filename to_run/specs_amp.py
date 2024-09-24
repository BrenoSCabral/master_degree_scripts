# -*- coding: utf-8 -*-
__doc__ = """
Created on Thu Sep 28 16:13:35 2023

Script para geração de espectros, convertido pelo ChatGPT do script matlab do Afonso

%  ----------------------------------------------------------------
% | calculate spectrum of a time/space series using fft            |
% |                                                                |
% | call:                                                          |
% |       [hepya,fff,conflim] = specs(xxx,ppp,dt,win,smo,ci)       |
% |                                                                |
% | xxx = time (space) serie                                       |
% | ppp = number of points for analysis                            |
% | dt  = sampling interval                                        |
% | win = type o spectral window (will be applyed in time domain)  |
% |       0 = no window                                            |
% |       1 = Hanning                                              |
% |       2 = cossine tappered                                     |
% | smo = smooth spectrum                                          |
% |        1 = no smoothing                                        |
% |       >1 = # points for running average (odd)                  |
% |      999 = variable length ruuning average                     |
% | ci  = confidence level                                         |
% |       (e.g., ci=95 means 95% confidence interval)              |
% |                                                                |
% | Returns:                                                       |
% |    fff  = cicles per dt unit                                   |
% |    hepy = (xxx unit)**2 / (cicles per dt unit)                 |
% |    conflim = confidence limits matrix [chi clo chi1 clo1 med]  |
% |       chi  = upper confidence limit                            |
% |       clo  = lower confidence limit                            |
% |       chi1 = upper confidence interval                         |
% |       clo1 = lower confidence interval                         |
% |       med  = (chi1+clo1)/2                                     |
% |                                                                |
% | OBS: see helpspec.m  for methods, references and plot hints    |
% |      see exampspec.m for examples on how to use                |
% |                                                                |
% | developed by: Afonso Paiva                                     |
% |                                                                |
%  ----------------------------------------------------------------

@author: b06x
"""

import numpy as np
from scipy.stats import chi2
import xarray as xr
import matplotlib.pyplot as plt

def specs(xxx, ppp, dt, win, smo, ci):
    # Clear internal variables
    # hwin = None
    hxxx = np.zeros(ppp)
    # hyyy = np.zeros(ppp)
    # hepy = np.zeros(ppp)
    # hepya = np.zeros(ppp)
    # fff = np.zeros(ppp)
    chi = []
    clo = []
    chi1= []
    clo1= []
    med = []
    # chi = np.zeros(ppp)
    # clo = np.zeros(ppp)
    # clo1 = np.zeros(ppp)
    # chi1 = np.zeros(ppp)
    # med = np.zeros(ppp)
    # conflim = np.zeros((ppp, 5))

    # Create spectral window
    len_xxx = len(xxx)
    winsize = min(ppp, len_xxx)

    if win == 0:
        hwin = np.ones(winsize)
    elif win == 1:
        hwin = np.hanning(winsize)
    elif win == 2:
        pi_val = np.pi
        wid = 0.1
        hwin = np.zeros(winsize)
        for i in range(winsize):
            if i <= wid * winsize:
                hwin[i] = 0.5 * (1 - np.cos(5 * pi_val * i / winsize))
            elif i >= (winsize - wid * winsize + 1):
                hwin[i] = 0.5 * (1 + np.cos(5 * pi_val * (i - 1) / winsize))
            else:
                hwin[i] = 1

    xxx = xxx.reshape(-1, 1)
    hwin = hwin.reshape(-1, 1)

    ########### CALCULATE SPECTRUM USING FFT ###########
    # Remove mean
    xxx = xxx - np.nanmean(xxx)

    # Apply window smoothing
    hxxx[:winsize] = hwin[:winsize].flatten() * xxx[:winsize].flatten()
    # hxx2 = hwin[:winsize]*xxx[:winsize]

    # Get number of harmonics
    nhar = (ppp + 1) // 2 if ppp % 2 != 0 else ppp // 2 + 1

    # Calculate Spectrum
    # Discrete Fourier Transform
    hyyy = np.fft.fft(hxxx)
    # Power Spectral Density
    hepy0 = np.abs(hyyy) ** 2 / len(hyyy) * dt

    hepy = hepy0[:nhar]

    # Window correction
    if win == 1:
        hepy[1:nhar] = 2.6 * hepy[1:nhar]
    elif win == 2:
        hepy[1:nhar] = 1.14 * hepy[1:nhar]

    ########### SMOOTH SPECTRUM ###########
    # No smoothing
    if smo == 1:
        hepya = hepy
    # Variable smoothing weights
    elif smo == 999:
        # Variable smoothing weights
        smo1, int_val, inc = 4, 10, 2
        smo1a, ind, int1 = smo1, smo1, int_val
        # i = 1 + smo1
        i = smo1
        hepya=[]
        while i < nhar - smo1:
            aux1 = np.sum(hepy[i - smo1:i + smo1+1] * np.hamming(2 * smo1 + 1))
            aux2 = np.sum(np.hamming(2 * smo1 + 1))
            hepya.append(aux1 / aux2)
            flag = 0
            if i >= int1:
                smo1 = smo1 + inc
                int1 = int1 + int_val
                flag = 1
            i = i + 1
        hepya=np.array(hepya)
        # while i <= nhar - smo1:
        #     print(i)
        #     aux1 = np.sum(hepy[i - smo1:i + smo1+1] * np.hamming(2 * smo1 + 1))
        #     aux2 = np.sum(np.hamming(2 * smo1 + 1))
        #     hepya[i - ind] = aux1 / aux2

        #     flag = 0
        #     if i >= int1:
        #         smo1 = smo1 + inc
        #         int1 = int1 + int_val
        #         flag = 1
        #     i = i + 1

        if flag == 1:
            smo1 = smo1 - inc
    else:
        # Constant smoothing weights
        # smo1 = (smo - 1) // 2
        # for i in range(smo1, nhar - smo1+1):
        #     print(i)
        #     hepya[i - smo1] = np.sum(hepy[i - smo1:i + smo1+1] * np.hamming(smo)) / np.sum(np.hamming(smo))
        hepya=[]
        for i in range(0,nhar-smo):
            hepya.append(np.sum(hepy[i:i+smo] * np.hamming(smo)) / np.sum(np.hamming(smo)))
        hepya=np.array(hepya)

    ########### CALCULATE CONFIDENCE INTERVAL ###########
    # Get time window factor for degrees of freedom
    if win == 0:
        wtfac = 1
    elif win == 1:
        wtfac = 1
    elif win == 2:
        # wid = 0.1
        wtfac = 1 - 2 * wid

    # Width correction factor for Hamming frequency smoothing
    wffac = 0.63
    alpha = 1 - ci / 100

    if smo == 999:
        x1, x2, aux = 0, int_val - 1, smo1a
        while x1 < len(hepya):
            # print(x1)
            # df = round(2 * (2 * aux + 1) * wtfac * wffac)
            # chi[x1:x2+1] = df * hepya[x1:x2+1] / chi2.ppf(alpha / 2, df)
            # clo[x1:x2+1] = df * hepya[x1:x2+1] / chi2.ppf(1 - alpha / 2, df)
            # chi1[x1:x2+1] = chi[x1:x2+1] / hepya[x1:x2+1]
            # clo1[x1:x2+1] = clo[x1:x2+1] / hepya[x1:x2+1]
            # med[x1:x2+1] = np.arange(x1, x2+1) / (x2 - x1 + 1)
            # x1 = x2 + 1
            # x2 = x2 + int_val
            # aux = aux + inc
            # if x2 >= len(hepya):
            #     x2 = len(hepya) - 1
            # print(x1)
            df = round(2 * (2 * aux + 1) * wtfac * wffac)
            chi[x1:x2+1] = df * hepya[x1:x2+1] / chi2.ppf(alpha / 2, df)
            clo[x1:x2+1] = df * hepya[x1:x2+1] / chi2.ppf(1 - alpha / 2, df)
            # chi1[x1:x2+1] = chi[x1:x2+1] / hepya[x1:x2+1]
            # clo1[x1:x2+1] = clo[x1:x2+1] / hepya[x1:x2+1]
            med[x1:x2+1] = np.arange(x1, x2+1) / (x2 - x1 + 1)
            x1 = x2 + 1
            x2 = x2 + int_val
            aux = aux + inc
            if x2 >= len(hepya):
                x2 = len(hepya) - 1
        chi = np.array(chi)
        clo = np.array(clo)
        chi1 = np.array(chi1)
        chi1 = chi/hepya
        clo1 = clo/hepya
        med = np.array(med)
    else:
        df = round(2 * smo * wtfac * wffac)
        chi = df * hepya / chi2.ppf(alpha / 2, df)
        clo = df * hepya / chi2.ppf(1 - alpha / 2, df)
        chi1 = chi / hepya
        clo1 = clo / hepya
        med = np.arange(len(hepya)) / len(hepya)

    confliml = [ chi, clo, chi1, clo1, med]
    conflim = np.array(confliml)


    ########### CALCULATE FREQUENCY/WAVE NUMBER AXIS ###########
    # Frequency array
    lll = len(hepya)
    fn = 1/(2*dt)
    if smo==1:
        fff = fn * np.arange(lll)/(ppp/2)
    elif smo==999:
        fff = (1 / (2 * dt)) * np.arange(lll) / (ppp / 2)
    else:
        fff = (1 / (2 * dt)) * np.arange(lll) / (ppp / 2)

    # hepya = hepya.reshape(-1, 1)
    # fff = fff.reshape(-1, 1)

    return fff, hepya, conflim


def spec_sum(da, **kw):
    __doc__= """
    parâmetros de entrada para a rotina de cálculo do espectro.
    Dado um Datarray da, extrai:
        xxx: numpy array
        ppp: comprimento do array
        dt: intervalo da série
    kwargs:
        win = 1 # Tipos de janela de filtragem:
                # win=0: retangular
                # win=1: hanning
                # win=2: consine
        smo= 31 # Número de pontos da janela de filtragem no domínio da frequência
    """
    win=kw.pop('kw',1)
    smo=kw.pop('smo',31)
    ci=kw.pop('ci',95)

    xxx = da.values
    ppp = len(xxx)  # Comprimento da janela de filtro no domínio do tempo
                    # (por enquanto vamos usar o comprimento da série)
    try:
        dt = (da.time.to_series(). # tava index no lugar de time
            diff().astype('timedelta64[m]').
            fillna(0).astype('int')/60)[1] # Intervalo amostral (inferindo do Dataframe df)
    except Exception:
        dt = (da.index.to_series(). 
            diff().astype('timedelta64[m]').
            fillna(0).astype('int')/60)[1] # Intervalo amostral (inferindo do Dataframe df)

    if kw.pop('hourly'):
        dt == dt/24      
    
    # abaixo eh quando eu jogo o array ao inves do dataframe

    # xxx= da
    # dt = 1
    # ppp = 365

    # ci = 95 # Intervalo de confiança (a verificar)

    fff, hepya, conflim = specs(xxx, ppp, dt, win, smo, ci)
    return fff, hepya, conflim


def sing_plot(hepyao, chio, cloo, prao, image_path):

# log log
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'Densidade Espectral [$m^2/dia$]')
    ax.set_xlabel(r'Período [dias]')

    ax.loglog(prao,hepyao, label = 'DADO', linewidth=1, color='red')
    ax.fill_between(prao,cloo,chio,alpha=0.2, color='tab:red')
    ax.legend(loc='lower right')
    ax.grid(which='major')
    ax.grid(which='minor',color='lightgrey')

    plt.tight_layout()
    plt.savefig(image_path + 'spectral_anal_loglog.png', dpi=200, bbox_inches='tight')
    plt.close()


def double_plot(hepyao_1, chio_1, cloo_1, prao_1, hepyao_2, chio_2, cloo_2, prao_2, label1, label2, image_path):
    # log log
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    ax.set_ylabel(r'Densidade Espectral [$m^2/dia$]')
    ax.set_xlabel(r'Período [dias]')

    ax.loglog(prao_1,hepyao_1, label = label1, linewidth=1, color='red')
    ax.fill_between(prao_1,cloo_1,chio_1,alpha=0.2, color='tab:red')

    ax.loglog(prao_2,hepyao_2, label = label2, linewidth=1, color='blue')
    ax.fill_between(prao_2,cloo_2,chio_2,alpha=0.2, color='tab:blue')

    ax.legend(loc='lower right')
    ax.grid(which='major')
    ax.grid(which='minor',color='lightgrey')

    plt.tight_layout()
    plt.savefig(image_path + 'spectral_anal_loglog.png', dpi=200, bbox_inches='tight')
    plt.close()


def spec_anal(serie, image_path, hourly=False):
    # getting result: (data must be an xarray)
    # for instance:     
    # data_raw_xr = xr.DataArray(np.asarray(data['ssh']), 
    # coords={'time': data.index}, 
    # dims=["time"])

    fffo, hepyao, conflim = spec_sum(serie, smo=999, win=1, hourly=hourly)
    fffo, hepyao = fffo[1:], hepyao[1:]
    chio = conflim[0]; chio=chio[1:]
    cloo = conflim[1]; cloo=cloo[1:]
    prao=1/fffo
    sing_plot(hepyao, chio, cloo, prao, image_path)


def double_spec_anal(serie_1, serie_2, image_path, label1, label2):
    fffo_1, hepyao_1, conflim_1 = spec_sum(serie_1, smo=999, win=1)
    fffo_1, hepyao_1 = fffo_1[1:], hepyao_1[1:]
    chio_1 = conflim_1[0]; chio_1=chio_1[1:]
    cloo_1 = conflim_1[1]; cloo_1=cloo_1[1:]
    prao_1=1/fffo_1

    fffo_2, hepyao_2, conflim_2 = spec_sum(serie_2, smo=999, win=1)
    fffo_2, hepyao_2 = fffo_2[1:], hepyao_2[1:]
    chio_2 = conflim_2[0]; chio_2=chio_2[1:]
    cloo_2 = conflim_2[1]; cloo_2=cloo_2[1:]
    prao_2=1/fffo_2



    double_plot(hepyao_1, chio_1, cloo_1, prao_1, hepyao_2, chio_2, cloo_2, prao_2, label1, label2, image_path)