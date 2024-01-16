  function [hepya,fff,conflim] = specs(xxx,ppp,dt,win,smo,ci)

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


% *******************  CLEAR INTERNAL VARIABLES  *******************
clear hwin hxxx hyyy hepy hepya fff chi clo

% ********************  CREATE SPECTRAL WINDOW  ********************

% -------------- length of window
len = length(xxx);
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

xxx = xxx(:);
hwin = hwin(:);

% ********************  CALCULATE SPECTRUM USING FFT  ********************

% --------------------  remove average
xxx = xxx - mean(xxx);
 
% --------------------  apply window
hxxx(1:winsize) = hwin(1:winsize).*xxx(1:winsize);

% -------------------- get number of harmonics
if rem(ppp,2),         %% ppp is odd
  nhar = (ppp+1)/2;
else                   %% ppp is even
  nhar = ppp/2+1;
end

% --------------------  calculate spectrum
hyyy = fft(hxxx,ppp);
hepy = hyyy.*conj(hyyy)/length(hyyy) * dt;
hepy(nhar+1:ppp) = [];
hepy(2:nhar) = 2*hepy(2:nhar);

% --------------------  Window correction
if win == 1                           
  hepy(2:nhar) = 2.6*hepy(2:nhar);
elseif win == 2
  hepy(2:nhar) = 1.14*hepy(2:nhar);
end

% **************************  SMOOTH SPECTRUM  ***********************

% --------------------  no smoothging
if smo == 1;

hepya = hepy;

% --------------------  variable smoothing weights
elseif smo == 999;

%  smo1=2; int=10; inc=1;
  smo1=4; int=10; inc=2;

smo1a=smo1; ind=smo1; int1=int;
i = 1+smo1;
while i <= nhar-smo1;
 aux1 = sum(hepy(i-smo1:i+smo1).*hamming(2*smo1+1)');
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
 
% --------------------  constant smoothing weights
else

 smo1 = (smo-1)/2;
 for i = 1+smo1:nhar-smo1;
  hepya(i-smo1) = sum(hepy(i-smo1:i+smo1).*hamming(smo)')/sum(hamming(smo));
 end

end

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

if smo == 999;

 x1=1; x2=int; aux=smo1a;
 while x1 < length(hepya);
%   df = 2 * (2*aux+1);
   df = round( 2 * (2*aux+1) * wtfac * wffac) ;
   chi(x1:x2) = df .* hepya(x1:x2) / chi2inv(  alpha/2,df);
   clo(x1:x2) = df .* hepya(x1:x2) / chi2inv(1-alpha/2,df);
   x1=x2+1; x2=x2+int; aux=aux+inc;
   if x2 > length(hepya); x2=length(hepya); end
 end
 chi1 = chi./hepya;
 clo1 = clo./hepya;

else

%   df = 2 * (2*aux+1);
 df = round( 2 * smo * wtfac * wffac );
 chi = df .* hepya / chi2inv(  alpha/2,df);
 clo = df .* hepya / chi2inv(1-alpha/2,df);
 chi1 = chi./hepya;
 clo1 = clo./hepya;

end

med = (1:length(hepya))./(1:length(hepya));
conflim = [ chi(:) clo(:) chi1(:) clo1(:) med(:)];

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

% ************ MAKE SURE ALL VECTORS ARE COLUMN VECTORS *************

hepya = hepya(:);
fff = fff(:);

return

