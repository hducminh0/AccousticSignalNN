
%--------------------------------------------------------------------------------
% Fonction signal. 
% Elle a pour objectif de d�finir une forme temporelle de signal impusionnel
% Et d'en caculer la transform�e de Fourier pour obtenir les amplitudes complexes
% Des diff�rentes composantes fr�quentielles. 
%
% Fait par Ghita et Emmanuel le 29 novembre 2012
%--------------------------------------------------------------------------------

function [frequ, coeff] = SignalFunction(f, nbpt, t0, tf) ;

%clear all ;
% Initialisation
%--------------------------------------------------------------------------------
%f = 100e6 ; 					 	% Fr�quence centrale du capteur 
%nbpt = 1e3 ; 						% Nombres de points temporels
%t0 = 0 ; % -100e-2 ; 			% Temps initial
%tf = 1e-7 ; % 00e-3 ;			% Temps final
Deltat = (tf-t0)/(nbpt-1) ;	% Delta temps 
t = t0:Deltat:tf ;				% Vecteur temps
Ampl = 1 ;							% Amplitude du signal

A=1 ;
sigma = 3e0 ;
C=0 ;
D=0 ;
w=2 ;

w = 2*pi*f ; 						% Pulsation temporelle
% D�finition du signal temporel
%--------------------------------------------------------------------------------
s = Ampl*sin(w*t) ; 				% Sinus � la fr�quence centrale

Env= A*exp(-(-w*(t-0.1e-7)).^2/(2*sigma^2))+D ; % enveloppe

signaltemp = Env.*s ;

% Calcul de la FFT
%--------------------------------------------------------------------------------
% nbfft = 1024*128 ;
nbfft = 1024;
long = length(signaltemp) ;
while(long>nbfft)
   nbfft = nbfft*2 ;
end

if (long<nbfft)
   % Zero padding
   t(long+1:nbfft) = t0 + Deltat*(long+1:nbfft) ;
   signaltemp((long+1:nbfft))= zeros(nbfft-long,1)  ;
end

% figure 
% plot(t, (signaltemp)) ;

%---------------------------------------------------------------------------------------------
% Calcul des spectre de puissance de signaltemp

fmax = 1/Deltat ;
Deltaf = 1/(t(length(t))-t(1)) ;

%freq = (-fmax/2:Deltaf:fmax/2).';   
freq = (0:Deltaf:fmax).';   

fminaff = 2.5e3 ;
fmaxaff = min(25e8, fmax/2) ;


%index0 = find(((-fmaxaff<freq) & (freq<-fminaff))) ;
%fMax = max(freq) ;
index0 = find((fmaxaff>freq) & (freq>fminaff)) ;
%index0 = find(((fMax-fmaxaff<freq) & (freq<fMax-fminaff))) ;
%index0 = find(((fmaxaff>freq) & (freq>fminaff)) & ((fMax-fmaxaff<freq) & (freq<fMax-fminaff))) ;
%index0 = find(((-fmaxaff<freq) & (freq<-fminaff)) | ((freq>fminaff) & (fmaxaff>freq))) ;
%index0 = find(((-fmaxaff<freq) & (freq<-fminaff)) | ((freq>fminaff) & (fmaxaff>freq))) ;

%keyboard

frequ = freq(index0)' ;
yy = fft(signaltemp, nbfft);
%yy = fftshift(yy) ;

%test = zeros(1, length(freq)) ;
% test(index0) = yy(index0) ;
% 
% figure
% plot(freq(1: length(yy)),abs(yy)) ;
% hold on
% %keyboard
% plot(freq(index0),abs(yy(index0)), 'LineWidth', 5) ;


coeff = yy(index0) ;

size(coeff);

%keyboard
