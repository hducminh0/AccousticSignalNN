% -------------------------------------------------------------------------
% Programme de calcul des formes d'ondes obtenues lors d'une mesure
% par microscopie acoustique.
% Application au cas de l'étude des tubes - These Hajar.
% -------------------------------------------------------------------------
clear all
close all

% -------------------------------------------------------------------------
% Choix du cas à traiter
% -------------------------------------------------------------------------
% cas = 1 ;

% switch cas
%     case 1 % Focalisation en surface
%         % -----------------------------------------------------------------
%         % Pz27 Aluminium Tungstene
%         % -----------------------------------------------------------------
%         % Epaisseurs
        % d = [0 3.210e-3 0] ;
        % d = [0 345e-6 0] ;
        % d = [0 300e-6 0] ;
%         % Densites
%         rho = [7700 2800 19254] ;
%         % Vitesses des ondes longitudinales dans les couches
%         vl = [4324 6420 5200] ;
%         % Vitesses des ondes transverses dans les couches
%         vt = [1659 3100 2900] ;
% end
for s = 1:1
   
    s
	data.thickness(s) = (round(rand() * 999) + 1)  * 10^-6
	d = [0 data.thickness(s) 0] ;
	% Densites
	rho = [7700 2800 19254] ;
	% Vitesses des ondes longitudinales dans les couches
	vl = [4324 6420 5200] ;
	% Vitesses des ondes transverses dans les couches
	vt = [1659 3100 2900] ;
	d = fliplr(d) ;
	rho = fliplr(rho) ;
	vl = fliplr(vl) ;
	vt = fliplr(vt) ;

	Ncouches = length(d);


	% -------------------------------------------------------------------------
	% Calcul des signaux
	% -------------------------------------------------------------------------

	for ncouches = Ncouches:Ncouches
	    % tic
	    
	    fc = round(200 - 200*rand()*0.15) * 1e6 ; 					% Fréquence centrale du capteur
	    
	    nbpts = 65536 ; 				% Nombres de points temporels
	    ti = 0 ; % -100e-2 ; 			% Temps initial
	    tfi = 5e-6 ; % 00e-3 ;			% Temps final
	    deltat = (tfi-ti)/(nbpts-1) ;	% Delta temps
	    tt = ti:deltat:tfi ;				% Vecteur temps
	    
	    % ---------------------------------------------------------------------
	    % Calcul des coefficients de la s�rie de Fourier dans la Bande Passante
	    % ---------------------------------------------------------------------
	    if (1) % Calcul du signal à partir de la bande passante
	        [frequence, coefficient] = SignalFunction(fc, nbpts, ti, tfi) ;
	    else % Chargement d'un signal expérimental
	        load signalfreq.mat;
	    end
	    
	    Trans = zeros(1, length(tt)) ;      % Coefficients de transmission
	    Refl = zeros(1, length(tt)) ;       % Coefficients de réflexion
	    
	    % toc
	    
	    w=2*pi*frequence' ;
	    % ---------------------------------------------------------------------
	    % Boucle fréquentielle
	    % ---------------------------------------------------------------------
	    % tic
	    for fr=1:length(frequence) % Pour chaque fréquence
	        %fr
	        %w = 2*pi*frequence(fr) ;
	        
	        k = w(fr)./vl ;
	        kappa = w(fr)./vt ;
	        
	        for itheta = 1:1 % Pour chaque angle d'incidence
	            angles(itheta) = 0.001 ; % (itheta-1)/180*pi ;
	            sigma = k(1)*sin(angles(itheta)) ;
	            
	            alpha = sqrt(k.^2-sigma^2) ;
	            beta = sqrt(kappa.^2-sigma^2) ;
	            Z = rho*w(fr)./alpha ;
	            P = alpha.*d ;
	            
	            Q = beta.*d ;
	            E = alpha/sigma ;
	            F = beta/sigma ;
	            G = 2*sigma^2./kappa.^2 ;
	            H = rho*w(fr)/sigma ;
	            
	            a(1,1,:) = G.*cos(P)+(1-G).*cos(Q) ;
	            a(1,2,:) = i*((1-G).*(sin(P)./E))-i*F.*G.*sin(Q) ;
	            a(1,3,:) = -(1./H).*(cos(P)-cos(Q)) ;
	            a(1,4,:) = -(i./H).*(sin(P)./E+F.*sin(Q)) ;
	            a(2,1,:) = i*E.*G.*sin(P)-i*((1-G).*sin(Q))./F ;
	            a(2,2,:) = (1-G).*cos(P) + G.*cos(Q) ;
	            a(2,3,:) = -(i./H).*(E.*sin(P)+sin(Q)./F) ;
	            a(2,4,:) = a(1,3,:) ;
	            a(3,1,:) = -H.*G.*(1-G).*(cos(P)-cos(Q)) ;
	            a(3,2,:) = -(i*H).*(((1-G).^2.*sin(P))./E  + F.*G.^2.*sin(Q)) ; %
	            a(3,3,:) = a(2,2,:) ;
	            a(3,4,:) = a(1,2,:) ;
	            a(4,1,:) = -i*H.*(E.*G.^2.*sin(P)+(1-G).^2.*sin(Q)./F) ;
	            a(4,2,:) = a(3,1,:) ;
	            a(4,3,:) = a(2,1,:) ;
	            a(4,4,:) = a(1,1,:) ;
	            
	            A=eye(4) ;
	            
	            for ieye = 2:ncouches-1
	                A = a(:,:,ieye)*A ;
	            end
	            
	            M(2,2) = A(2,2) - A(2,1)*A(4,2)/A(4,1) ;
	            M(2,3) = A(2,3) - A(2,1)*A(4,3)/A(4,1) ;
	            M(3,2) = A(3,2) - A(3,1)*A(4,2)/A(4,1) ;
	            M(3,3) = A(3,3) - A(3,1)*A(4,3)/A(4,1) ;
	            
	            T(itheta,fr) = 2*Z(1)/(M(3,2)+Z(1)*M(3,3)+(M(2,2)+Z(1)*M(2,3))*Z(ncouches)) ;
	            R(itheta,fr) = (M(3,2)+Z(1)*M(3,3)-(M(2,2)+Z(1)*M(2,3))*Z(ncouches))/(M(3,2)+Z(1)*M(3,3)+(M(2,2)+Z(1)*M(2,3))*Z(ncouches)) ;
	            
	        end
	    end
	    % toc
	    
	    sigfreqtrans = coefficient.*T(itheta,:) ;
	    sigfreqref = coefficient.*R(itheta,:) ;
	    
	    % Calcul de la IFFT
	    %--------------------------------------------------------------------------------
	    % nbfft = 1024*128 ;
	    nbfft = 1024;
	    % figure
	    % plot(frequence)
	    long = length(coefficient) ;
	    while(long>nbfft)
	        nbfft = nbfft*2 ;
	    end
	    
	    
	    
	    if (long<nbfft)
	        % Zero padding
	        frequen = frequence(length(frequence)) + (1:(nbfft-long))*(frequence(2)-frequence(1)) ;
	        frequence = [frequence frequen] ;
	        
	        sigfreqtrans = [conj(sigfreqtrans) zeros(1,nbfft-2*long) (fliplr(sigfreqtrans))] ;
	        sigfreqref = [conj(sigfreqref) zeros(1,nbfft-2*long) (fliplr(sigfreqref))] ;
	    end
	    % figure 
	    % plot(abs(sigfreqref)) ;
	    % figure
	    % plot(frequence)
	    % %keyboard
	    % %---------------------------------------------------------------------------------------------
	    % Calcul du signal temporel
	    
	    Tmax = 1/abs(frequence(2)-frequence(1)) ;
	    %Deltat = 1/(2*abs(frequence(1))) ;
	    Deltat = 1/(frequence(length(frequence))) ;
	    temps = (0:Deltat:2*Tmax);
	    
	    signaltrans = ifft(sigfreqtrans);
	    signalref = ifft(sigfreqref);
	end
	data.acc_signal(s, :) = awgn(real(signalref), 50);
%     data.acc_signal(s, :) = real(signalref);
% 	Temps = (1:length(signalref))*Deltat*1e6 ;
% 	figure
% 	plot(Temps, real((signalref))/max(real(signalref)), 'b', 'LineWidth', 2) ;
% 	hold on
% 	plot(Temps, imag((signalref))/max(real(signalref)), 'r', 'LineWidth', 2) ;
end
% data.thickness = data.thickness';
% save('data_10000_50_vf.mat', 'data');
