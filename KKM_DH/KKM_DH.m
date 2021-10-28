%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% High bandwidth utilization digital holographic multiplexing: an approach using Kramers¨CKronig relations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Citation for this code and algorithm:
% Zhengzhong Huang and Liangcai Cao
% "High bandwidth utilization digital holographic multiplexing: an approach using Kramers¨CKronig relations",
% Advanced Photonics Research
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code is written by Zhengzhong Huang, 2021
% The version of Matlab for this code is R2016b

%%
clc
clear
close all;

%% Reading pictures

amplitude_image = im2double(imread('Amplitude1.png'));% Simulation raw data
amplitude_image = amplitude_image(21:420,21:420);
amplitude_image = (amplitude_image./(max(amplitude_image(:))))+0.2;
amplitude_image = (amplitude_image./(max(amplitude_image(:))));
[m,n]=size(amplitude_image);

phase_image = im2double(imread('Phase1.jpg'));% Simulation raw data
phase_image = phase_image(21:420,21:420);
phase_image = 1.*(phase_image./max(phase_image(:)));


amplitude_image2 = im2double(imread('Amplitude2.jpg'));% Simulation raw data
amplitude_image2 = rgb2gray(amplitude_image2);
amplitude_image2 = amplitude_image2(21:420,21:420);
amplitude_image2 = (amplitude_image2./(max(amplitude_image2(:))))+0.2;
amplitude_image2 = (amplitude_image2./(max(amplitude_image2(:))));

phase_image2 = im2double(imread('Phase2.jpg'));% Simulation raw data
phase_image2 = rgb2gray(phase_image2);
phase_image2 = phase_image2(21:420,21:420);
phase_image2 = 1.*(phase_image2./max(phase_image2(:)));


%% Defined pupil function of coherent optical transfer function
radius = round(m/8)-1;% radius of pupil function
[x1,y2] = meshgrid(1:m,1:m);    
x1 = x1 - floor(max(x1(:))/2+1); 
y2 = y2 - floor(max(y2(:))/2+1); 
pupil_function = (x1./radius).^2+(y2./radius).^2 <= 1;% pupil function

%% Defined objects
Amplitude_image = fftshift(fft2(ifftshift(amplitude_image)));
Amplitude_image = Amplitude_image.*pupil_function;
Phase_image = fftshift(fft2(ifftshift(phase_image)));
Phase_image = Phase_image.*pupil_function;

amplitude_image = fftshift(ifft2(ifftshift(Amplitude_image)));
amplitude_image = abs(amplitude_image)./max(abs(amplitude_image(:)))+0.1;
amplitude_image = abs(amplitude_image)./max(abs(amplitude_image(:)));% Amplitude of object 1
phase_image = fftshift(ifft2(ifftshift(Phase_image)));
phase_image = abs(phase_image)./max(abs(phase_image(:)));% Phase of object 1


Amplitude_image2 = fftshift(fft2(ifftshift(amplitude_image2)));
Amplitude_image2 = Amplitude_image2.*pupil_function;
Phase_image2 = fftshift(fft2(ifftshift(phase_image2)));
Phase_image2 = Phase_image2.*pupil_function;

amplitude_image2 = fftshift(ifft2(ifftshift(Amplitude_image2)));
amplitude_image2 = abs(amplitude_image2)./max(abs(amplitude_image2(:)))+0.1;
amplitude_image2 = abs(amplitude_image2)./max(abs(amplitude_image2(:)));% Amplitude of object 2

phase_image2 = fftshift(ifft2(ifftshift(Phase_image2)));
phase_image2 = abs(phase_image2)./max(abs(phase_image2(:)));% Phase of object 2

object1 = amplitude_image.*exp(2.5.*1i*phase_image);
object2 = amplitude_image2.*exp(2.5.*1i*phase_image2);

Object1 = fftshift(fft2(ifftshift(object1)));
Object2 = fftshift(fft2(ifftshift(object2)));
Object1 = Object1.*pupil_function;
Object2 = Object2.*pupil_function;
object1 = fftshift(ifft2(ifftshift(Object1)));
object2 = fftshift(ifft2(ifftshift(Object2)));
object1 = object1./max(max(abs(object1(:))));% Object 1 in Fig. 2(b)
object2 = object2./max(max(abs(object2(:))));% Object 2 in Fig. 2(c)

%% 
sap_spatial_freq1 = -(round(m/8));% Frequency shift of object 1
sap_spatial_freq11 = round(m/8);% Frequency shift of object 1
sap_spatial_freq2 = round(m/8);% Frequency shift of object 2
sap_spatial_freq22 = round(m/8);% Frequency shift of object 2
%Object1=fftshift(fft2(ifftshift(object1)));
%Object2=fftshift(fft2(ifftshift(object2)));
Object11 = circshift(Object1,[sap_spatial_freq1,0]);
Object22 = circshift(Object2,[sap_spatial_freq2,0]);
sample1 = fftshift(ifft2(ifftshift(Object11)));% Object 1 after Frequency shift
sample2 = fftshift(ifft2(ifftshift(Object22)));% Object 2 after Frequency shift

%%
Reference_field = zeros(m,m);% Spectrum of reference wave
reference_freq = -round(m/8);
Reference_field(floor(m/2+1),floor(m/2+1)+reference_freq) = ...
max(abs(sample1(:)))*length(sample1(:))*2;
reference_field = fftshift(ifft2(ifftshift(Reference_field)));% Reference wave

%% Generated hologram 
diffraction_field = sample1+sample2+reference_field;
hologram = abs(diffraction_field).^2;
hologram = hologram./abs(reference_field).^2;% Hologram in Fig. 2(h)
Hologram = fftshift(fft2(ifftshift(hologram)));% Spectrum of hologram in Fig. 2(i)
figure(1),imshow(log(abs(Hologram)),[]),title('Spectrum of hologram');

padding_number = 20;
normalization_const1 = length(Hologram(:)); % Normalization constant
Hologram = padarray(Hologram,[0,m*padding_number],'both');
normalization_const2 =length(Hologram(:));% Normalization constant
Hologram = Hologram/normalization_const1*normalization_const2;
hologram1 = fftshift(ifft2(ifftshift(Hologram)));

%% KKM-DH

log_hologram = log(hologram1);
real_part = log_hologram./2; % log(abs(I))=(1/2)*log(|1+(S1+S2)/R|)^2
imaginary_part = imag(hilbert(real(real_part).')).'; % Hilbert transform
retrieval_field = exp(real_part+1i*imaginary_part)-1; % (S1+S2)/R = exp(log((S1+S2)/R+1))-1

% remove the zero padding to match the size with the answer
Retrieval_field = fftshift(fft2(ifftshift(retrieval_field)));
Retrieval_field = circshift(Retrieval_field,[0,-m*padding_number]);
Retrieval_field = Retrieval_field(:,1:m);
retrieval_field = fftshift(ifft2(ifftshift(Retrieval_field)));
retrieval_field = retrieval_field.*reference_field;% (S1+S2) = R*exp(log((S1+S2)/R+1))-1

%% Reconstruction from KKM-DH
pupil_function0 = zeros(m,m);
pupil_function0(m./2-radius+1:m./2+radius,...
    m./2-radius+1:m./2+radius) = 1;% Filtering function with rectangle, circle can also perform in this.
Reconstruction1 = circshift(Retrieval_field,[-sap_spatial_freq1,-sap_spatial_freq11]);
Reconstruction1 = Reconstruction1.*pupil_function0;
reconstruction1 = fftshift(ifft2(ifftshift(Reconstruction1)));

Reconstruction2 = circshift(Retrieval_field,[-sap_spatial_freq2,-sap_spatial_freq22]);
Reconstruction2 = Reconstruction2.*pupil_function0;
reconstruction2 = fftshift(ifft2(ifftshift(Reconstruction2)));
reconstruction1 = reconstruction1./max(max(abs(reconstruction1(:))));% Reconstruction of object 1
reconstruction2 = reconstruction2./max(max(abs(reconstruction2(:))));% Reconstruction of object 2
MSE1abs = (sum(sum((abs(reconstruction1)-abs(object1)).^2)))./(sum(sum((abs(object1)-1).^2)));% MSE of amplitude of object 1
MSE1phase = (sum(sum((angle(reconstruction1)-angle(object1)).^2)))./(sum(sum((angle(object1)).^2)));% MSE of phase of object 1
MSE2abs = (sum(sum((abs(reconstruction2)-abs(object2)).^2)))./(sum(sum((abs(object2)-1).^2)));% MSE of amplitude of object 2
MSE2phase = (sum(sum((angle(reconstruction2)-angle(object2)).^2)))./(sum(sum((angle(object2)).^2)));% MSE of phase of object 2

%% Reconstruction from conventional off-axis filtering
Hologram2 = fftshift(fft2(ifftshift(hologram)));

Reconstruction3 = circshift(Hologram2,[-sap_spatial_freq1,-sap_spatial_freq11]);
Reconstruction3 = Reconstruction3.*pupil_function0;
reconstruction3 = fftshift(ifft2(ifftshift(Reconstruction3)));

Reconstruction4 = circshift(Hologram2,[-sap_spatial_freq2,-sap_spatial_freq22]);
Reconstruction4 = Reconstruction4.*pupil_function0;
reconstruction4 = fftshift(ifft2(ifftshift(Reconstruction4)));
reconstruction3 = reconstruction3./max(max(abs(reconstruction3(:))));% Reconstruction of object 1
reconstruction4 = reconstruction4./max(max(abs(reconstruction4(:))));% Reconstruction of object 2
MSE3abs = (sum(sum((abs(reconstruction3)-abs(object1)).^2)))./(sum(sum((abs(object1)-1).^2)));% MSE of amplitude of object 1
MSE3phase = (sum(sum((angle(reconstruction3)-angle(object1)).^2)))./(sum(sum((angle(object1)).^2)));% MSE of phase of object 1
MSE4abs = (sum(sum((abs(reconstruction4)-abs(object2)).^2)))./(sum(sum((abs(object2)-1).^2)));% MSE of amplitude of object 2
MSE4phase = (sum(sum((angle(reconstruction4)-angle(object2)).^2)))./(sum(sum((angle(object2)).^2)));% MSE of phase of object 2

% Conparsion of reconstructed amplitude
figure(2),subplot(1,3,1),imshow(abs(object1),[]),title('Ground truth');
figure(2),subplot(1,3,2),imshow(abs(reconstruction1),[]),title('KKM-DH');
figure(2),subplot(1,3,3),imshow(abs(reconstruction3),[]),title('Directly filtering');

% Conparsion of reconstructed phase
figure(3),subplot(1,3,1),imshow(angle(object1),[]),title('Ground truth');
figure(3),subplot(1,3,2),imshow(angle(reconstruction1),[]),title('KKM-DH');
figure(3),subplot(1,3,3),imshow(angle(reconstruction3),[]),title('Directly filtering');


