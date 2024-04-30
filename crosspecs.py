import numpy as np

def crospecs(xx1, xx2, ppp, dt, win, smo, ci):
    # calculate cross spectrum of time/space series using fft
    # Arguments:
    # xx1: 1st time (space) serie
    # xx2: 2nd time (space) serie
    # ppp: number of points for analysis
    # dt: sampling interval
    # win: type o spectral window (will be applyed in time domain)
    #      0 = no window
    #      1 = Hanning
    #      2 = cossine tappered
    # smo: smooth spectrum (Hamming window)
    #      1 = no smoothing
    #      >1 = # points for running average (odd)
    #      999 = variable length ruuning average
    # ci: confidence level
    #     (e.g., ci=95 means 95% confidence interval)
    
    # Returns:
    # fff: cicles per dt unit
    # spectra: (xx1 unit)**2 / (cicles per dt unit)
    # hepya1: spectrum of first serie
    # hepya2: spectrum of second serie
    # coef: coherence coefficient
    # conf: confidence interval for coef
    # fase: phase of 2nd serie in relation to 1st
    
    # Internal variables:
    # hwin: spectral window in time domain
    # hepya1: spectrum of first serie
    # hepya2: spectrum of second serie
    # hepyac: cross spectrum
    # hepycra: real part of cross spectrum
    # hepycia: imaginary part of cross spectrum
    # OBS: same variables without "a" are not yet smoothed
    # ************  CREATE SPECTRAL WINDOW ********************
    # -------------- length of window
    tamanho = len(xx1)
    if tamanho > ppp:
        winsize = ppp
    elif tamanho <= ppp:
        winsize = tamanho

    hwin = np.zeros((np.size(xx1)))
    hhh = np.zeros((np.size(xx1)))
    # --------------- no window
    if win == 0:
        for i in range(0,winsize):
            hwin[i] = 1
    # ------------------- hanningning window
    elif win == 1:
        hwin = np.hanning(winsize);

    elif win == 2:
         pi  = 3.14159
         wid = 0.1
         for i in range(winsize):
            if i <= wid * winsize:
                hwin[i] = 0.5 * (1 - np.cos(5 * np.pi * i / winsize))
            elif i >= (winsize - wid * winsize + 1):
                hwin[i] = 0.5 * (1 + np.cos(5 * np.pi * (i - 1) / winsize))
            else:
                hwin[i] = 1

    hhh[1:winsize+1] = hwin[1:winsize+1]

    # ************ MAKE SURE ALL VECTORS ARE COLUMN VECTORS *************
    xx1 = np.reshape(xx1, (-1, 1))
    xx2 = np.reshape(xx2, (-1, 1))
    hhh = np.reshape(hhh, (-1, 1))

    # ******************** COMPUTE SPECTRA **************************
    # --------------------  remove average
    xx1 = xx1 - np.mean(xx1)
    xx2 = xx2 - np.mean(xx2)

    # --------------------  apply window
    xx1[:winsize] = hhh[:winsize] * xx1[:winsize]
    xx2[:winsize] = hhh[:winsize] * xx2[:winsize]

    # --------------------  calculate spectrum for both series
    for jj in range(0,2):
        if jj == 0:
            zzz = xx1
        elif jj == 1:
            zzz = xx2
        
        hyyy = np.fft.fft(zzz[:,0], ppp)
        hepy = hyyy * np.conj(hyyy) / len(hyyy) * dt
        hepy = hepy[0:ppp//2+2]# corte adicionado para correção ****
        # hepy[ppp//2+1:] = []#cortando o espectro
        hepy[1:ppp//2+1] = 2*hepy[1:ppp//2+1]

        # Window correction
        if win == 1:
            hepy[1:ppp//2+1] = 2.6 * hepy[1:ppp//2+1]
        elif win == 2:
            hepy[1:ppp//2+1] = 1.14 * hepy[1:ppp//2+1]

        if jj == 0:
            hepy1 = hepy
            cru1 = np.conj(hyyy)
        elif jj == 1:
            hepy2 = hepy
            cru2 = hyyy

    # ******************* COMPUTE CROSS SPECTRA **********************
    hepy = cru1*cru2/len(cru1) * dt
    hepy = hepy[0:ppp//2+2]
    hepy[1:ppp//2+1] = 2*hepy[1:ppp//2+1]

    # Window correction
    if win == 1:
        hepy[1:ppp//2+1] = 2.6*hepy[1:ppp//2+1]
    elif win ==2:
        hepy[1:ppp//2+1] = 1.14*hepy[1:ppp//2+1]

    hepyc = hepy
    hepycr = np.real(hepyc)
    hepyci = np.imag(hepyc)

    # -------------  make sure all vectors are column vectors
    hepy1 = hepy1.reshape(-1, 1)
    hepy1=hepy1[:,0]
    hepy2 = hepy2.reshape(-1, 1)
    hepy2=hepy2[:,0]
    hepyc = hepyc.reshape(-1, 1)
    hepyc=hepyc[:,0]
    hepycr = hepycr.reshape(-1, 1)
    hepycr=hepycr[:,0]
    hepyci = hepyci.reshape(-1, 1)
    hepyci=hepyci[:,0]

    # *********************** SMOOTH SPECTRA *************************
    # -----------------------  no smoothing
    if smo == 1:
        hepya1 = hepy1
        hepya2 = hepy2
        hepyac = hepyc
        hepycra = hepycr
        hepycia = hepycr
    # -----------------------  smoothing
    else:
        # --- for each variable
        for jj in range(0, 5):
            if jj == 0:
                hepy = hepy1
            elif jj == 1:
                hepy = hepy2
            elif jj == 2:
                hepy = hepyc
            elif jj == 3:
                hepy = hepycr
            elif jj == 4:
                hepy = hepyci

            # ---  variable smoothing weights
            if smo == 999:
                smo1 = 2
                int_val = 4
                inc = 1
                smo1a = smo1
                ind = smo1
                int1 = int_val
                i = 0 + smo1
                hepya=[]
                while i <= ppp//2 + 1 - smo1:
                    aux1 = np.sum(np.real(hepy[i - smo1:i + smo1 + 1]) * np.hamming(2 * smo1 + 1))
                    aux2 = np.sum(np.hamming(2 * smo1 + 1))
                    # hepya[i - ind] = aux1 / aux2
                    hepya.append(aux1/aux2)
                    flag = 0

                    if i >= int1:
                        smo1 = smo1 + inc
                        int1 = int1 + int_val
                        flag = 1
                    
                    i = i + 1
                
                # transformar hepya em array
                hepya=np.asarray(hepya)

                if flag == 1:
                    smo1 = smo1 - inc

            # ---  constant smoothing weights
            else:
                smo1 = (smo - 1) // 2
                hepya=[]
                for i in range(1 + smo1, ppp//2 + 1 - smo1):
                    aux1 = np.sum(np.real(hepy[i - smo1 -1:i + smo1 + 1]) * np.hamming(smo))
                    aux2 = np.sum(np.hamming(smo))
                    # hepya[i - smo1] = aux1 / aux2
                    hepya.append(aux1 / aux2)
                # transformar hepya em array
                hepya=np.asarray(hepya)

            # ---  finish smoothing one variable
            if jj == 0:
                hepya1 = hepya
            elif jj == 1:
                hepya2 = hepya
            elif jj == 2:
                hepyac = hepya
            elif jj == 3:
                hepycra = hepya
            elif jj == 4:
                hepycia = hepya

            # --- go to next variable
            # ---  finish smoothing

    # ****************** COMPUTE CONFIDENCE INTERVAL *******************
    # -------------  get time window factor for degrees of freedom
    coef = (hepycra**2 + hepycia**2) / (hepya1 * hepya2)

    #  ****************** COMPUTE CONFIDENCE INTERVAL *******************
    # -------------  get time window factor for degrees of freedom
    if win == 0:
        wtfac = 1
    elif win == 1:
        wtfac = 1
    elif win == 2:
        wtfac = 1 - 2 * wid

    # -----------  width correction factor for Hamming freq. smoothing
    wffac = 0.63
    alpha = 1 - ci/100

    if smo == 999:
        int1=int_val
        aux=smo1a
        conf=[]
        for i in range(len(hepya)):
            # df = 2 * (2*aux+1)
            df = round(2 * (2*aux+1) * wtfac * wffac)
            conf.append(1 - alpha**(1/(df/2-1)))
            if i == int1:
                int1 += int_val
                aux += inc
        conf=np.asarray(conf)
    else:
        df = round(2 * smo * wtfac * wffac)
        jj = np.arange(1, len(hepya)+1)
        conf = (1 - alpha**(1/(df/2-1))) * jj / jj


    # ******** COMPUTE PHASE OF 2nd SERIES IN RELATION TO 1st *********
    fase = np.arctan2(-hepycia, hepycra)

    # ***************  CALCULATE FREQUENCY/WAVE NUMBER AXIS **************
    fn = 1 / (2 * dt)
    lll = len(hepya)
    if smo == 1:
        fff = fn * np.arange(lll) / (ppp / 2)
    elif smo == 999:
        fff = fn * np.arange(smo1a, lll + smo1a) / (ppp / 2)
    else:
        fff = fn * np.arange(smo1, lll + smo1) / (ppp / 2)

    return hepya1,hepya2,fff,coef,conf,fase
