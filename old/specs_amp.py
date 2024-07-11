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
    dt = (da.time.to_series(). # tava index no lugar de time
          diff().astype('timedelta64[m]').
          fillna(0).astype('int')/60)[1] # Intervalo amostral (inferindo do Dataframe df)
    
    # abaixo eh quando eu jogo o array ao inves do dataframe

    # xxx= da
    # dt = 1
    # ppp = 365

    # ci = 95 # Intervalo de confiança (a verificar)

    fff, hepya, conflim = specs(xxx, ppp, dt, win, smo, ci)
    return fff, hepya, conflim

def plot():
    # da=dfobs['u'][~np.isnan(dfobs['u'])] -> IMPORTA DADO
            # da=dfobs['u'].interpolate('linear')
    # fffo, hepyao, conflim = spa.spec_sum(da,smo=999,win=1) # CALCULA ESPECTRO
    fffo, hepyao, conflim = spec_sum(data/100,smo=999,win=1)
    fffo, hepyao = fffo[1:], hepyao[1:]
    chio = conflim[0]; chio=chio[1:]
    cloo = conflim[1]; cloo=cloo[1:]
    # chio = conflim[2]/1000; chio=chio[1:]
    # cloo = conflim[3]/1000; cloo=cloo[1:]
    prao=1/fffo
    # Espectro do modelo
    # da=dfmod['u']
    spao_fft = {}
    for spao in spaos:
        fffm, hepyam , conflim = spec_sum(spaos[spao],smo=999,win=1)
        fffm, hepyam = fffm[1:], hepyam[1:]
        chim = conflim[0]; chim=chim[1:]
        clom = conflim[1]; clom=clom[1:]
        pram=1/fffm
        spao_fft[spao] = [fffm, hepyam, chim, clom, pram]

    # --------- Gráfico -------
    # from matplotlib.ticker import ScalarFormatter, NullFormatter
    # import matplotlib.ticker as mticker

    # fig=plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    # ax.set_xlabel(r'Período [dias]')
    # ax.set_ylabel(r'Densidade Espectral de Potência [$m^2/dia$]')
    # ax = fig.add_subplot(111)
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    # ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    # ax1=fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)

    # axis = [ax1, ax2, ax3, ax4]

    # for axe, spao in zip(axis, spaos):
    #     axe.loglog(prao,hepyao, label = 'dados', linewidth=1)
    #     axe.fill_between(prao,cloo,chio,alpha=0.2, color='tab:blue')

    #     axe.loglog(spao_fft[spao][-1],spao_fft[spao][1], label = spao, linewidth=1)
    #     axe.fill_between(spao_fft[spao][-1],spao_fft[spao][-2],spao_fft[spao][2],alpha=0.2, color='tab:orange')
    #     axe.legend(loc='lower right')
    #     axe.grid(which='major')
    #     axe.grid(which='minor',color='lightgrey')
    # # ax1.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10]))

    # # ax1.set_xticks([20, 55])
    # # ax1.set_yticks([20, 55])


    # # plt.title('Componente zonal - '+loc + ' z = '+ str(zp))
    # # plt.grid(which='major')
    # # plt.grid(which='minor',color='lightgrey')

# NOVO GRAFICO

    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylabel(r'Densidade Espectral [$m^2/dia$]')
    ax.set_xlabel(r'Período [dias]')

    axes = [fig.add_subplot(241), fig.add_subplot(242), fig.add_subplot(245), fig.add_subplot(246)]
    colors={'HYCOM': 'tab:orange', 'GLOR4':'tab:blue', 'GLOR12':'tab:purple', 'BRAN':'tab:grey'}
    for axis, spao in zip(axes, spaos):
        axis.semilogx(prao,hepyao, label = 'DADO', linewidth=1, color='red')
        axis.fill_between(prao,cloo,chio,alpha=0.2, color='tab:red')
        axis.semilogx(spao_fft[spao][-1],spao_fft[spao][1], label = spao, linewidth=1, color=colors[spao][4:])
        axis.fill_between(spao_fft[spao][-1],spao_fft[spao][-2],spao_fft[spao][2],alpha=0.2, color=colors[spao])
        axis.legend(loc='lower right')
        axis.grid(which='major')
        axis.grid(which='minor',color='lightgrey')

    plt.tight_layout()
    plt.savefig(image_path + 'spectral_anal_semilog.png', dpi=200, bbox_inches='tight')
    plt.close()

    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.set_ylabel(r'Densidade Espectral [$m^2/dia$]')
    ax.set_xlabel(r'Período [dias]')

    axes = [fig.add_subplot(241), fig.add_subplot(242), fig.add_subplot(245), fig.add_subplot(246)]
    colors={'HYCOM': 'tab:orange', 'GLOR4':'tab:blue', 'GLOR12':'tab:purple', 'BRAN':'tab:grey'}
    for axis, spao in zip(axes, spaos):
        axis.loglog(prao,hepyao, label = 'DADO', linewidth=1, color='red')
        axis.fill_between(prao,cloo,chio,alpha=0.2, color='tab:red')
        axis.loglog(spao_fft[spao][-1],spao_fft[spao][1], label = spao, linewidth=1, color=colors[spao][4:])
        axis.fill_between(spao_fft[spao][-1],spao_fft[spao][-2],spao_fft[spao][2],alpha=0.2, color=colors[spao])
        axis.legend(loc='lower right')
        axis.grid(which='major')
        axis.grid(which='minor',color='lightgrey')

    plt.tight_layout()
    plt.savefig(image_path + 'spectral_anal_dilog.png', dpi=200, bbox_inches='tight')
    plt.close()

    # ax1=fig.add_subplot(121)
    # # ax1.loglog(prao,hepyao, label = 'dados', linewidth=1)
    # # ax1.fill_between(prao,cloo,chio,alpha=0.2, color='tab:blue')
    # ax1.loglog(prao,hepyao, label = 'DADO', linewidth=1, color='red')
    # ax1.fill_between(prao,cloo,chio,alpha=0.2, color='tab:red')
    # colors={'HYCOM': 'tab:orange', 'GLOR4':'tab:blue', 'GLOR12':'tab:purple', 'BRAN':'tab:grey'}
    # for spao in spaos:
    #     ax1.loglog(spao_fft[spao][-1],spao_fft[spao][1], label = spao, linewidth=1, color=colors[spao][4:])
    #     ax1.fill_between(spao_fft[spao][-1],spao_fft[spao][-2],spao_fft[spao][2],alpha=0.2, color=colors[spao])
    # ax1.legend(loc='lower center')
    # ax1.set_xlabel(r'Período [dias]')
    # ax.set_ylabel(r'Densidade Espectral de Potência [$m^2/dia$]')
    # # plt.title('Componente zonal - '+loc + ' z = '+ str(zp))
    # plt.grid(which='major')
    # plt.grid(which='minor',color='lightgrey')


    # fig=plt.figure(figsize=(10,5))
    # ax1=fig.add_subplot(121)
    # ax1.loglog(prao,hepyao, label = 'dados', linewidth=1)
    # ax1.fill_between(prao,cloo,chio,alpha=0.2, color='tab:blue')
    # ax1.loglog(pram,hepyam, label = 'REMO LSE24', linewidth=1)
    # ax1.fill_between(pram,clom,chim,alpha=0.2, color='tab:orange')
    # ax1.legend(loc='lower right')
    # ax1.set_xlabel(r'Período [h]')
    # ax1.set_ylabel(r'Densidade Espectral de Potência [$m^2/s^2.h$]')
    # # plt.title('Componente zonal - '+loc + ' z = '+ str(zp))
    # plt.grid(which='major')
    # plt.grid(which='minor',color='lightgrey')


def sing_plot(hepyao, chio, cloo, prao, image_path):
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    # ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
    ax.set_ylabel(r'Densidade Espectral [$m^2/dia$]')
    ax.set_xlabel(r'Período [dias]')

    ax.semilogx(prao,hepyao, label = 'DADO', linewidth=1, color='red')
    ax.set_xlim(0,25)
    ax.fill_between(prao,cloo,chio,alpha=0.2, color='tab:red')
    ax.legend(loc='lower right')
    ax.grid(which='major')
    ax.grid(which='minor',color='lightgrey')

    plt.tight_layout()
    plt.savefig(image_path + 'spectral_anal_semilog.png', dpi=200, bbox_inches='tight')
    plt.close()


# log log
    fig=plt.figure(figsize=(20,10))
    ax = fig.add_subplot(121)
    # ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # ax.spines['top'].set_color('none')
    # ax.spines['bottom'].set_color('none')
    # ax.spines['left'].set_color('none')
    # ax.spines['right'].set_color('none')
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


def main():
    # getting result: (data must be an xarray)
    # for instance:     
    # data_raw_xr = xr.DataArray(np.asarray(data['ssh']), 
    # coords={'time': data.index}, 
    # dims=["time"])

    fffo, hepyao, conflim = spec_sum(data_raw_xr/100,smo=999,win=1)
    fffo, hepyao = fffo[1:], hepyao[1:]
    chio = conflim[0]; chio=chio[1:]
    cloo = conflim[1]; cloo=cloo[1:]
    prao=1/fffo