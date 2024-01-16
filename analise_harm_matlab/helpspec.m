
%  ------------------------------------------------------------------
% |                                                                  |
% |                     SPECS.M  --  CROSPECS.M                      |
% |                                                                  |
% |            HELP FILE FOR SPECTRAL ANALYSIS PACKAGE               |
% |                                                                  |
% |                         Afonso Paiva                             |
% |                       (Juin 6th, 1997)                           |
%  ------------------------------------------------------------------
% |                                                                  |
% |    The programs specs.m and crospecs.m do spectral and cross     |
% |    spectral analysis of equaly spaced time series. They are not  |
% |    supposed to be complete or even perfect. They just do it in   |
% |    a nice way that was suitable for my purposes.                 |
% |    Older versions spec.m and crospec.m do a bad smoothing        |
% |                                                                  |
%  ------------------------------------------------------------------
% |                                                                  
% | 0) CALLS                                                                 
% | [hepya,fff,conflim] = specs(xxx,ppp,dt,win,smo,ci)
% | [hepya1,hepya2,fff,coef,conf,fase] =  
% |                      crospecs(xx1,xx2,ppp,dt,win,smo,ci) 
% |                                                                  
% | 1) INPUT / OUTPUT
% |    xxx = time series for spectral analysis
% |    xx1 = 1th time series for cross spectral analysis
% |    xx2 = 2nd time series for cross spectral analysis
% |    ppp = number of points for analysis
% |    dt  = sampling interval
% |    win = time window type
% |    smo = control spectra smoothing
% |    ci  = confidence level
% | 
% |    hepya  = spectrum of time series for specs
% |    hepya1 = spectrum of 1st time series for crospecs
% |    hepya2 = spectrum of 2st time series for crospecs
% |    fff    = frequencies
% |    conflim = confidence limits for spectrum
% |    coef    = coherence squared
% |    conf    = confidence limit for coherence
% |    fase    = phase for crospec
% |
% | 2) TIME WINDOW
% |    Applyed in time domain to reduce syde lobs leakage. 
% |    Options are: win=0 (no window), win=1 (Hanning window) and 
% |                 win=2 (cossine tappered window) 
% |    For win=2 can control size of window editing value of wid.
% |    Default is 10% (wid=0.1)
% |    
% | 3) SPECTRUM and CROSS SPECTRUM
% |    Computed using FFT of whole series. Can have number of points
% |    for analysis (ppp) as the same of time series, or increase to
% |    next power of 2 (fill with zeros) or any value.
% |    Computed using MATLAB FFT routine as: 
% |       spectrum:       Gxx(f) = (2.dt/N)*([Xk]+)^2
% |       cross spectrum: Gxy(f) = (2.dt/N)*[Xk*.Yk]+
% |    Note that MATLAB FFT does not include (dt), and that is why
% |    it appears in the numerator. For a FFT routine with (dt), the
% |    spectrum would be
% |        Gxx(f) = 2/(dt.N).([Xfk]+)^2
% |    and including (dt^2) gives previous expression
% |    Spectrum unit is: 
% |       [variable unit square divided by cicles per tiem unit]
% |    Final variable is hepy, which length is (ppp/2+1)
% | 
% | 4) TIME WINDOW CORRECTION
% |    Applied to spectrum and cross spectrum to correct for energy
% |    loss due to time window: 
% |    1.14 for cossine tappered and 2.6 for Hanning
% | 
% | 5) SMOOTHING
% |    Running average smoothing of spectra to reduce variance and
% |    increase statistical confidence, or reduce confidence limits.
% |    Use a Hamm window - a box window would introduce spurious 
% |    picks and phase problems.
% |    Get to compromising between strong smothing (more confidence
% |    but sronger bias) and weak smoothing (less confidence but
% |    less bias) - See Ref.b
% |    Two options are avaiable:
% |    a) Fixed window size (smo>1)
% |    b) Variable window size (smo=999). 
% |       The idea is to have smaller size for low frequencies,
% |    where have less points, and progressively increase size to 
% |    higher frequencies. Will have weaker smoothing for lower
% |    frequencies, with les statistical confidence, but will not
% |    lose information.
% |    If smo=999, than the width (smo=2*smo1+1) is increased by 
% |    (inc) at every interval (int). Default values are
% |    smo1=2; int=10; inc=1;                                     
% |    To control these values edit this line inside the program.
% | 
% | 6) RESOLUTION
% |    Raw spectrum has resolution of B=1/T. Smoothed spectrum has
% |    resolution B=smo/T.
% |    Note that although B increases, program compute spectrum for
% |    all the frequencies of raw spectrum.
% | 
% | 7) DEGREES OF FREEDOM
% |    Used in computation of confident limits. Most simple
% |    definition is: 
% |       df=2*smo
% |    Here I use the one used by Rainer Zanttopp, where:
% |       df = 2 * smo * tap * wffac
% |    with (wffac=0.63) a correction for the effective size of the
% |    Hamming window and (tap=1-2*wid) a correction for the time 
% |    windowing with cossine tappered for first and last 100*wid%
% |    of the data. For Hannig window I used (tap=1) as I just do
% |    not know which correction to apply.
% |    See references a, e h, i
% | 
% | 8) COHERENCE SQUARED
% |    Given by:
% |       C2 = (C^2+Q^2) / (Gxx.Gyy)
% |    where (Gxy=C+Qi)
% | 
% | 9) CONFIDENCE LIMITS
% |    For spectrum, given by (ref. d , pg 286):
% |         {df/[Chi(df,a/2)]} <= s2 <= {df/[Chi(df,1-a/2)]}
% |    where Chi is Chi square distribution, df is number of degrees
% |    freedom and (a=1-p/100) for p=confidence level (e.g., 95%, 90%).
% |    For coherence, given by (refs f,g):
% |         c2 = 1 - a**[1/(df/2-1)]
% |    OBS: If compute coherence (square root of C2), then has to use
% |         square root of c2.
% | 
% | 10)PHASE 
% |    Given by : f = atan(-Q/C) for Gxy=C+Qi, represents the 
% |    phase of 2nd serie in relation to 1st 
% | 
% | 11)PLOTS
% | 
% |    a) Spectra
% |       plot(fff,hepya)
% |       semilogx(fff,hepya)
% |       loglog(fff,hepya)
% |       semilogx(fff,hepya1/max(hepya1),fff,hepya2/max(hepya2))
% |       plotyy(fff,hepya1,'r',fff,hepya2,'g',[1 0 0 0 0.5 0 .2 0 100])
% |       semilogx(fff,hepya1.*fff*2.3)
% | 
% |    b) Coherence and phase
% |       semilogx(fff,coef,fff,conf)
% |       semilogx(fff,fase)
% | 
% |    c) Confidence limits
% |       semilogx(fff,chi,fff,clo)
% |       loglog(fff,med*sc,fff,chi1*sc,fff,clo1*sc)
% |           where sc is ordinate of med
% | 
% | 
% |    d) confidence interval in loglog
% |       sc=0.01;chi1=sc*conflim(:,3);clo1=sc*conflim(:,4);
% |       f1=1;loglog(fff,hepya),hold on,line([f1 f1],[clo1 chi1])
% |       line([f1-0.001 f1+0.001],[chi1 chi1])
% |       line([f1-0.001 f1+0.001],[clo1 clo1])  % testar volores somados a f1
% |       obs: f1 posiciona linha nas abcissas e sc posiciona nas ordenadas
% |
% | 12)REFERENCES
% | 
% |    a) My notes
% |    
% |    b) Mello Filho, E. (1982). Investigacao sobre a analise da
% |       agitacao maritima, Theses, COPPE-UFRJ.
% |    
% |    c) Processamento e analise de sinais. Relatorio, PETROBRAS.
% |    
% |    d) Bendat, J.S and A.G. Piersol (1986). Random data - analysis
% |       and measurement procedures. John Willey & Sons.
% |    
% |    e) Jenkins, G.M and D.G. Watts (1968). Spectral analysis and
% |       its applications, Holden-day, S. Francisco.
% |    
% |    f) Julian, P.R. (1975). Comments on the determination of 
% |       significance level of the coherence statistics, Journal
% |       of the Atmospheric Sciences, 32, 836-837.
% |    
% |    g) Thompson, R.O.R.Y (1979) Coherence significance levels.
% |       Journal of the Atmospheric Sciences, october, 2020-2021.
% |    
% |    h) Cooley, Lewis and Welch (1967) The FFT algorithm and its
% |       applications, DOC RC1743, IBM Research, (pg 139).
% |    
% |    i) NCAR routine specft.for
% |    
%  -----------------------------------------------------------------------      
