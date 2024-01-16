% ****************************************************************
% ****************************************************************
% *****                                                      *****
% *****      EXAMPLES FOR USING spec.m and crossspec.m       *****
% *****                                                      *****
% ****************************************************************
% ****************************************************************

clear
clg

% ********** GETTING DATA

% suppose file dataspec with 3 columns
% get 2nd column as x1 and 3th column as x2
 load ggg
 x1 = ggg(:,2);
 x2 = ggg(:,3);

% or just create vectors x1 and x2
%t = 1:100;
%x1 = sin(2*pi*t/50) + 0.7*sin(2*pi*t/10) + randn(size(t));
%x2 = sin(2*pi*t/45) + 0.6*sin(2*pi*t/10) + randn(size(t));

% ****************************************************************
% ********** SPECTRAL ANALYSIS OF x1
% ****************************************************************

ppp = 512;       % # points as power of 2
dt = 1;          % sampling interval
win = 1;         % Hanning spectral window
smo = 7;         % running average with 7 points
ci = 95;         % 95 percent confidence interval

[hepy,fff,chi,clo] = spec(x1,ppp,dt,win,smo,ci);

% ********** PLOT RESULTS

clg

% simple energy spectra
subplot(3,2,1)
plot(fff,hepy)

% loglog plot
subplot(3,2,2)
loglog(fff,hepy)

% x log scale plot
subplot(3,2,3)
semilogx(fff,hepy)

% variance conserving spectra
a = hepy.*fff;
subplot(3,2,4)
semilogx(fff,a)

% confidence intervals
subplot(3,2,5)
plot(fff,hepy)
hold
plot(fff,chi,'r')
plot(fff,clo,'r')

% confidence intervals in log scale
xs(1) = 1e-2;
xs(2) = 1e-2;
dif(1) = 1e-2;
dif(2) = log(chi(1))-log(clo(1));
subplot(3,2,6)
loglog(fff,hepy)
hold
plot(xs,dif,'g')


% ****************************************************************
% ********** CROSS SPECTRAL ANALYSIS
% ****************************************************************

figure

ppp = 512;       % # points as power of 2
dt = 1;          % sampling interval
win = 1;         % Hanning spectral window
smo = 7;         % running average with 7 points
ci = 95;         % 95 percent confidence level

[hepya1,hepya2,fff,coef,conf,fase] = crospec(x1,x2,ppp,dt,win,smo,ci);

% plot time series
subplot(3,2,1)
plot(x1)
subplot(3,2,2)
plot(x2)

% plot spectra
subplot(3,2,3)
plot(fff,hepya1)
subplot(3,2,4)
plot(fff,hepya2)

% plot coherence and confidence interval
subplot(3,2,5)
plot(fff,coef)
hold
dif(1:length(fff)) = conf;
plot(fff,dif,'g')

%plot fase
subplot(3,2,6)
plot(fff,fase)




