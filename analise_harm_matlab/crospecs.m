function[hepya1,hepya2,fff,coef,conf,fase] = crospecs(xx1,xx2,ppp,dt,win,smo,ci)

%  ----------------------------------------------------------------
% | calculate cross spectrum of time/space series using fft        |
% |                                                                |
% | call:                                                          |
% |        [hepya1,hepya2,fff,coef,conf,fase] =                    |
% |                      crospecs(xx1,xx2,ppp,dt,win,smo,ci)       |
% |                                                                |
% | xx1 = 1st time (space) serie                                   |
% | xx2 = 2nd time (space) serie                                   |
% | ppp = number of points for analysis                            |
% | dt  = sampling interval                                        |
% | win = type o spectral window  (will be applyed in time domain) |
% |       0 = no window                                            |
% |       1 = Hanning                                              |
% |       2 = cossine tappered                                     |
% | smo = smooth spectrum (Hamming window)                         |
% |        1 = no smoothing                                        |
% |       >1 = # points for running average (odd)                  |
% |      999 = variable length ruuning average                     |
% | ci  = confidence level                                         |
% |       (e.g., ci=95 means 95% confidence interval)              |
% |                                                                |
% | Returns: fff  = cicles per dt unit                             |
% |          spectra : (xx1 unit)**2 / (cicles per dt unit)        |
% |              hepya1 - spectrum of first serie                  |
% |              hepya2 - spectrum of second serie                 |
% |          coef = coherence coefficient                          |
% |          conf = confidence interval for coef                   |
% |          fase = phase of 2nd serie in relation to 1st          |
% |                                                                |
% | OBS: see helpspec.m for methods, references and plot hints     |
% |                                                                |
% | developed by: Afonso Paiva                                     |
% |                                                                |
%  ----------------------------------------------------------------


% *********************** INTERNAL VARIABLES ***********************

% hwin    = spectral window in time domain
% hepya1  = spectrum of first serie
% hepya2  = spectrum of second serie
% hepyac  = cross spectrum
% hepycra = real part of cross spectrum
% hepycia = imaginary part of cross spectrum
% OBS: same variables without "a" are not yet smoothed

% *******************  CLEAR INTERNAL VARIABLES  *******************

clear hwin hepya1 hepya2 hepyac hepycra hepycia hepya1n hepya2n coef fase

% ********************  CREATE SPECTRAL WINDOW  ********************

% ------------------ length of window
len = length(xx1);
if len > ppp
  winsize = ppp;
elseif len <= ppp
  winsize = len;
end

% ------------------- no window
if win == 0
  for i = 1:winsize
    hwin(i)= 1;
  end

% ------------------- Hanning window
elseif win == 1
  hwin = hanning(winsize);

% ------------------- cosine-tapered applied to first and last 10% of data
elseif win == 2
 pi  = 3.14159;
 wid = 0.1;
 for i = 1:winsize;
   if i <= wid*winsize
     hwin(i) = 0.5 * (1-cos(5*pi*i/winsize));
   elseif i >= (winsize-wid*winsize+1)
     hwin(i) = 0.5 * (1+cos(5*pi*(i-1)/winsize));
   else
     hwin(i) = 1;
   end
 end

end
hhh(1:winsize) = hwin(1:winsize)';

% ************ MAKE SURE ALL VECTORS ARE COLUMN VECTORS *************

xx1 = xx1(:);
xx2 = xx2(:);
hhh = hhh(:);

% ******************** COMPUTE SPECTRA **************************

% --------------------  remove average
xx1 = xx1 - mean(xx1);
xx2 = xx2 - mean(xx2);

% --------------------  apply window

xx1(1:winsize) = hhh(1:winsize).*xx1(1:winsize);
xx2(1:winsize) = hhh(1:winsize).*xx2(1:winsize);

% --------------------  calculate spectrum for both series
for jj = 1:2;
 
  if jj==1;, zzz = xx1;
  elseif jj==2;, zzz = xx2;
  end

  hyyy = fft(zzz,ppp);
  hepy = hyyy.*conj(hyyy)/length(hyyy) * dt;
  hepy(ppp/2+2:ppp) = [];
  hepy(2:ppp/2) = 2*hepy(2:ppp/2);
 
% Window correction
  if win == 1;
    hepy(2:ppp/2) = 2.6*hepy(2:ppp/2);
  elseif win == 2; 
    hepy(2:ppp/2) = 1.14*hepy(2:ppp/2);
  end
 
  if jj==1;, hepy1 = hepy;, cru1 = conj(hyyy);
  elseif jj==2;, hepy2 = hepy;, cru2 = hyyy;
  end
 
end

% ******************* COMPUTE CROSS SPECTRA **********************

hepy = cru1.*cru2/length(cru1) * dt;
hepy(ppp/2+2:ppp) = [];
hepy(2:ppp/2) = 2*hepy(2:ppp/2);

% Window correction
if win == 1;
  hepy(2:ppp/2) = 2.6*hepy(2:ppp/2);
elseif win == 2;
  hepy(2:ppp/2) = 1.14*hepy(2:ppp/2);
end

hepyc  = hepy;
hepycr = real(hepyc);
hepyci = imag(hepyc);

% -------------  make sure all vectors are column vectors
hepy1  = hepy1(:);
hepy2  = hepy2(:);
hepyc  = hepyc(:);
hepycr = hepycr(:);
hepyci = hepyci(:);

% *********************** SMOOTH SPECTRA *************************

% -----------------------  no smoothing
if smo == 1;

  hepya1  = hepy1;
  hepya2  = hepy2;
  hepyac  = hepyc;
  hepycra = hepycr;
  hepycia = hepycr;

% -----------------------  smoothing
else

% --- for each variable
  for jj = 1:5;

    if jj==1;,     hepy = hepy1;
    elseif jj==2;, hepy = hepy2;
    elseif jj==3;, hepy = hepyc;
    elseif jj==4;, hepy = hepycr;
    elseif jj==5;, hepy = hepyci;
    end

% ---  variable smoothing weights
    if smo == 999;

      smo1=4; int=10; inc=1;
%      smo1=4; int=20; inc=1;
%      smo1=5; int=20; inc=2;

      smo1a=smo1; ind=smo1; int1=int;
      i = 1+smo1;
      while i <= ppp/2+1-smo1;
        aux1 = sum(hepy(i-smo1:i+smo1).*hamming(2*smo1+1));
        aux2 = sum(hamming(2*smo1+1));
        hepya(i-ind) = aux1 / aux2;
        flag=0;
        if i >= int1;
          smo1 = smo1 + inc;
          int1 = int1 + int;
          flag=1;
        end
        i = i + 1;
      end
      if flag == 1;
        smo1=smo1-inc;
      end;

% ---  constant smoothing weights
    else

      smo1 = (smo-1)/2;
      for i = 1+smo1:ppp/2+1-smo1;
       aux1 = sum(hepy(i-smo1:i+smo1).*hamming(smo));
       aux2 = sum(hamming(smo));
       hepya(i-smo1) = aux1 / aux2;
      end

% ---  finish smoothing one variable
    end

    if jj==1;,     hepya1 = hepya;
    elseif jj==2;, hepya2 = hepya;
    elseif jj==3;, hepyac = hepya;
    elseif jj==4;, hepycra = hepya;
    elseif jj==5;, hepycia = hepya;
    end

% --- go to next variable
  end

% ---  finish smoothing
end


%  ************** COMPUTE SQUARED COHERENCE COEFFICIENT *************

coef = ( hepycra.^2 + hepycia.^2 ) ./ (hepya1.*hepya2);

%  ****************** COMPUTE CONFIDENCE INTERVAL *******************

% -------------  get time window factor for degrees of freedom
if win == 0;
  wtfac = 1;
elseif win ==1;
  wtfac = 1;
elseif win == 2;
  wtfac = 1 - 2*wid;
end

% -----------  width correction factor for Hamming freq. smoothing
wffac = 0.63;

alpha = 1 - ci/100;

if smo == 999;

 int1=int; aux=smo1a;
 for i = 1:length(hepya);
%   df = 2 * (2*aux+1);
   df = round( 2 * (2*aux+1) * wtfac * wffac );
   conf(i) = 1 - alpha^(1/(df/2-1));
   if i == int1;
     int1=int1+int;
     aux=aux+inc;
   end
 end

else

%   df = 2 * (2*aux+1);
 df = round( 2 * smo *wtfac * wffac );
 jj=1:1:length(hepya);
 conf(1:length(hepya)) = (1 - alpha^(1/(df/2-1))) * jj./jj;

end


% ******** COMPUTE PHASE OF 2nd SERIE IN RELATION TO 1st *********

fase = atan2(-hepycia,hepycra);
 
% ***************  CALCULATE FREQUENCY/WAVE NUMBER AXIS **************

fn = 1/(2*dt);
lll=length(hepya);
if smo == 1;
  fff = fn * (0:lll-1)/(ppp/2);
elseif smo == 999;
  fff = fn * (smo1a:lll+smo1a-1)/(ppp/2);
else
  fff = fn * (smo1:lll+smo1-1)/(ppp/2);
end


return
