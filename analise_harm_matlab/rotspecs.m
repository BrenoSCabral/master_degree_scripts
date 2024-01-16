  function [hpos,hneg,htot,hrcoef,fff,conflim] = rotspecs(xxx1,xxx2,ppp,dt,win,smo,ci)

%  --------------------------------------------------------------------
% | calculate rotary spectra of [u,v] velocity time series using fft   |
% |                                                                    |
% | call:                                                              |
% | [hpos,hneg,htot,hrcoef,fff,conflim] =                              | 
% |                            specs(xxx1,xxx2,,ppp,dt,win,smo,ci)     |
% |                                                                    | 
% | xx1 = time serie of u (zonal)      velocity component              |
% | xx2 = time serie of v (meridional) velocity component              |
% | ppp = number of points for analysis                                |
% | dt  = sampling interval                                            |
% | win = type o spectral window (will be applyed in time domain)      |
% |       0 = no window                                                |
% |       1 = Hanning (I don't like much)                              |
% |       2 = cossine tappered (best)                                  |
% | smo = smooth spectrum                                              |
% |        1 = no smoothing                                            |
% |       >1 = # points for running average (odd)                      |
% |      999 = variable length ruuning average                         |
% |            editar specs e modificar parametros se necessario       |
% |            smo1 = No pesos                                         |
% |            int = intervalo (em pts para mudanca de pesos)          |
% |            inc = incremento da mudanca                             |
% |            amo1=No pesos; int = intervalo de mudança;              |
% | ci  = confidence level                                             |
% |       (e.g., ci=95 means 95% confidence interval)                  |
% |                                                                    |
% | Returns:                                                           |
% |    fff  = cicles per dt unit                                       |
% |    Energy density --> (xxx unit)**2 / (cicles per dt unit)         |
% |           hpos = positive part of the spectrum                     |
% |           hneg = negative part of the spectrum                     |
% |           htot = total spectrum                                    |
% |           hrcoef = coeficiente de rotacao                          |
% |    conflim = confidence limits matrix [chi clo chi1 clo1 med]      |    
% |       chi  = upper confidence limit                                |    
% |       clo  = lower confidence limit                                |    
% |       chi1 = upper confidence interval                             |    
% |       clo1 = lower confidence interval                             |    
% |       med  = (chi1+clo1)/2                                         |
% |                                                                    |
% | OBS: see helpspec.m  for methods, references and plot hints        |
% |      see exampspec.m for examples on how to use                    |
% |                                                                    |
% | developed by: Afonso Paiva                                         |
% | obs: valor de energia parece estar correto, mas...                 | 
% |      use por sua conta e risco                                     |      
% |                                                                    | 
%  --------------------------------------------------------------------


% *******************  CLEAR INTERNAL VARIABLES  *******************
clear hxxx1 hxx2 hyyy1 hyyy2 hepy1 hepy2 hepya_pos hepya_neg
clear hwin fff chi clo 
clear hpos hneg htot au bu av bv

% ********************  CREATE SPECTRAL WINDOW  ********************

% -------------- length of window
len = length(xxx1);
if len > ppp
  winsize = ppp;
elseif len <= ppp
  winsize = len;
end 

% --------------- no window
if win == 0
  hwin= ones(winsize,1);

% --------------- Hanning window
elseif win == 1
  hwin = hanning(winsize);

% --------------- cosine-tapered applied to first and last 100*wid% of data
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

% ************ MAKE SURE ALL VECTORS ARE COLUMN VECTORS *************

xxx1 = xxx1(:);
xxx2 = xxx2(:);
hwin = hwin(:);

% ****************  CALCULATE SPECTRUM USING FFT  *******************

% --------------------  remove average
xxx1 = xxx1 - mean(xxx1);
xxx2 = xxx2 - mean(xxx2);
 
% --------------------  apply window
hxxx1(1:winsize) = hwin(1:winsize).*xxx1(1:winsize);
hxxx2(1:winsize) = hwin(1:winsize).*xxx2(1:winsize);

% -------------------- get number of harmonics
if rem(ppp,2),         %% ppp is odd
  nhar = (ppp+1)/2;
else                   %% ppp is even
  nhar = ppp/2+1;
end

% --------------------  calculate spectrum
hyyy1 = fft(hxxx1,ppp); hepy1=hyyy1;
hepy1(nhar+1:ppp) = [];
hepy1(2:nhar) = 2*hepy1(2:nhar);
% hepy1(2:nhar) = hepy1(2:nhar);

hyyy2 = fft(hxxx2,ppp); hepy2 = hyyy2;
hepy2(nhar+1:ppp) = [];
hepy2(2:nhar) = 2*hepy2(2:nhar);
% hepy2(2:nhar) = hepy2(2:nhar);

% --------------------  Window correction
if win == 1                           
  hepy1(2:nhar) = 2.6*hepy1(2:nhar);
  hepy2(2:nhar) = 2.6*hepy2(2:nhar);
elseif win == 2
  hepy1(2:nhar) = 1.14*hepy1(2:nhar);
  hepy2(2:nhar) = 1.14*hepy2(2:nhar);
end

% *************** CALCULA ESPECTRO ROTATORIO ************************

% ---------- partes real e imaginaria
au=real(hepy1);bu=imag(hepy1);
av=real(hepy2);bv=imag(hepy2);

% ---------- espectros para rotacao positiva e negativa
hpos= ( (au+bv).^2+(av-bu).^2 ) /ppp /4 * dt;
hneg= ( (au-bv).^2+(av+bu).^2 ) /ppp /4 * dt;
%%%htot = hpos + hneg;


% **************************  SMOOTH SPECTRUM  ***********************

% --------------------  no smoothging
if smo == 1;

hepya_pos = hpos;
hepya_neg = hneg;
%%%hepya_tot = htot;

% --------------------  variable smoothing weights
elseif smo == 999;

%  smo1=2; int=10; inc=1;
  smo1=4; int=10; inc=1;

smo1a=smo1; ind=smo1; int1=int;
i = 1+smo1;
while i <= nhar-smo1;
 aux1 = sum(hpos(i-smo1:i+smo1).*hamming(2*smo1+1)');
 aux2 = sum(hamming(2*smo1+1));
        hepya_pos(i-ind) = aux1 / aux2;
 aux1 = sum(hneg(i-smo1:i+smo1).*hamming(2*smo1+1)');
 aux2 = sum(hamming(2*smo1+1));
        hepya_neg(i-ind) = aux1 / aux2;
%%% aux1 = sum(htot(i-smo1:i+smo1).*hamming(2*smo1+1)');
%%% aux2 = sum(hamming(2*smo1+1));
%%%        hepya_tot(i-ind) = aux1 / aux2;
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
 
% --------------------  constant smoothing weights
else

 smo1 = (smo-1)/2;
 for i = 1+smo1:nhar-smo1;
  hepya_pos(i-smo1) = sum(hpos(i-smo1:i+smo1).*hamming(smo)')/sum(hamming(smo));
  hepya_neg(i-smo1) = sum(hneg(i-smo1:i+smo1).*hamming(smo)')/sum(hamming(smo));
  %%%hepya_tot(i-smo1) = sum(htot(i-smo1:i+smo1).*hamming(smo)')/sum(hamming(smo));
 end

end

% ************ OBTEM PARAMETROS DERIVADOS DOS ESPECTROS **************

% ---------- renomea variaveis
hpos = hepya_pos;
hneg = hepya_neg;
%%%htot = hepya_tot;

% ---------- espectro total
htot = hpos + hneg;

% ---------- coeficiente de rotacao
hrcoef = (hpos-hneg)./htot;


% ****************** CALCULATE CONFIDENCE INTERVAL *******************

% -------------  get time window factor for degrees of freedom
if win == 0;
  wtfac = 1;
elseif win == 1;
  wtfac = 1;
elseif win == 2;
  wtfac = 1 - 2*wid;
end

% -----------  width correction factor for Hamming freq. smoothing
wffac = 0.63;

alpha = 1 - ci/100;

% ----------  intervalo de confianca para smooth variavel
if smo == 999;

 x1=1; x2=int; aux=smo1a;
 while x1 < length(htot);
   %%%%% df = 2 * (2*aux+1);
   df = round( 2 * (2*aux+1) * wtfac * wffac) ;
   chi(x1:x2) = df .* htot(x1:x2) / chi2inv(  alpha/2,df);
   clo(x1:x2) = df .* htot(x1:x2) / chi2inv(1-alpha/2,df);
   x1=x2+1; x2=x2+int; aux=aux+inc;
   if x2 > length(htot); x2=length(htot); end
 end
 chi1 = chi./htot;
 clo1 = clo./htot;

 % ----------  intervalo de confianca para os demais
else

 %%%%% df = 2 * (2*aux+1);
 df = round( 2 * smo * wtfac * wffac );
 chi = df .* htot / chi2inv(  alpha/2,df);
 clo = df .* htot / chi2inv(1-alpha/2,df);
 chi1 = chi./htot;
 clo1 = clo./htot;

end

med = (1:length(htot))./(1:length(htot));
conflim = [ chi(:) clo(:) chi1(:) clo1(:) med(:)];


% ********************************************************************
%%%%% obs fica aqui por enquanto
%%%%% apos = sqrt(hpos * ppp / dt);
%%%%% aneg = sqrt(hneg * ppp / dt);
%%%%% emax = apos + aneg;
%%%%% emin = abs(apos - aneg);

% ***************  CALCULATE FREQUENCY/WAVE NUMBER AXIS **************

fn = 1/(2*dt);
lll=length(htot);
if smo == 1;
  fff = fn * (0:lll-1)/(ppp/2);
elseif smo == 999;
  fff = fn * (smo1a:lll+smo1a-1)/(ppp/2);
else
  fff = fn * (smo1:lll+smo1-1)/(ppp/2);
end

% ************ MAKE SURE ALL VECTORS ARE COLUMN VECTORS *************

hpos = hpos(:);
hneg = hneg(:);
htot = htot(:);
hrcoef = hrcoef(:);
fff = fff(:);


return

