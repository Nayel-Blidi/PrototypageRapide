%% PART I
clear all
close all

% Parameters
Fs = 10e3; 
Ts = 1/Fs;
Rb = 2e3;
Ns = Fs/Rb*2; 
T = Ts*Ns;
Tb = 1/Rb;
M = 2; 
alpha = 0.35; 
fc = 2e3; 
EbN0dB = 0:1:8;
n_bits=2457600;
h = rcosdesign(0.35,8,Ns,'sqrt');
hr = fliplr(h);


% Message
%message_bits = randi([0, M-1], 1, n_bits);
nom_fichier = 'message.txt';
message_bits = load(nom_fichier);
message_bits = message_bits.';
%%
% Complexe symbols
symbols_mapped = pskmod(message_bits, M, pi/M,'gray');
    
%4-PSK Gray mapping plot
figure();
scatter(real(symbols_mapped), imag(symbols_mapped), 'x', 'red'), grid on
% close all
    
% Oversampling
symbols_oversampled = kron(symbols_mapped, [1 zeros(1,Ns-1)]);
% scatter(real(symbols_oversampled(1:100)), imag(symbols_oversampled(1:100)))
  
% Pulse shapping
retard = (length(h) - 1)/2;
symbols_pulse_shapped = filter(h, 1, [symbols_oversampled, zeros(1, retard)]);
 
t = linspace(0,(length(symbols_pulse_shapped)-1)/Fs,length(symbols_pulse_shapped));
% Plot Xe & PSD
figure()
subplot(2,1,1)
plot(real(symbols_pulse_shapped(1:100)))
subplot(2,1,2)  
plot(imag(symbols_pulse_shapped(1:100)))
figure()
DSP = pwelch(symbols_pulse_shapped,[],[],[],Fs,'twosided', 'centered');
semilogy(DSP)
title("DSP of the emitted signal x"), grid on
    
signal_exponential = symbols_pulse_shapped.*exp(1j*2*pi*fc*t);
signal_reel = real(signal_exponential);
   
%figure()
% plot(signal_reel(1:100))

sigma = 4/20;
noise = sqrt(sigma) * randn(1,length(signal_reel));
signal_reel_noisy = signal_reel + noise;
signal_exponential_rx = 2*signal_reel_noisy.*exp(-1j*2*pi*fc*t); % on multiplie par 2 pour avoir la bonne puissance
    
%close all
%figure
%scatter(real(signal_exponential_rx), imag(signal_exponential_rx))
    
% Reception filtering
signal_filtered = filter(hr, 1, [signal_exponential_rx zeros(1, retard)]);
%%
% Sampling
signal_sampled = signal_filtered(2*retard+1:Ns:end);
%figure()
%scatter(real(signal_sampled(1:1000)), imag(signal_sampled(1:1000)))
% Demapping
signal_demodulated = pskdemod(signal_sampled, M, pi/M, 'gray');
  
% Verification
erreur = sum(signal_demodulated ~= message_bits);



%% Projet Prototypage
% Nom du fichier dans lequel vous souhaitez enregistrer la variable
nom_fichier = 'D:/Documents/IPSA/A5/Prototypage/msg_recu_sigma4.txt';

% Ouverture du fichier en mode écriture
fid = fopen(nom_fichier, 'w');

% Écriture de la variable dans le fichier
fprintf(fid, '%d\n', signal_demodulated);

% Fermeture du fichier
fclose(fid);


