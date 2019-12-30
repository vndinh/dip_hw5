clear;

Problem_1();

function Problem_1()
% Template for EE535 Digial Image Processing
% Insert the code in the designated area below
%% Loading directory for image files
imgdir = uigetdir('D:\KAIST\Courses\dip\hw\hw5\Test_images');
file = fopen(fullfile(imgdir,'\monarch_gray_512x512.raw'),'rb');
gray_monarch = fread(file,fliplr([512,512]),'*uint8')';
fclose(file);
%%
%%---------------------Insert code below ----------------------%%
%% 1. Image Restoration
gray_monarch = double(gray_monarch);
hamming_monarch = HammingFilter(gray_monarch);
gauss_monarch = GaussianNoise(hamming_monarch,0.086);

% 1.1. Pseudo-Inverse
p_inv_monarch_1 = PseudoInvFilter(gauss_monarch,0.3);
p_inv_monarch_2 = PseudoInvFilter(gauss_monarch,0.5);
p_inv_monarch_3 = PseudoInvFilter(gauss_monarch,0.7);

psnr_p_inverse_1 = PSNR(gray_monarch,p_inv_monarch_1);
psnr_p_inverse_2 = PSNR(gray_monarch,p_inv_monarch_2);
psnr_p_inverse_3 = PSNR(gray_monarch,p_inv_monarch_3);

psnr_p_inverse = psnr_p_inverse_2;

% 1.2. Wiener Filter
wiener_monarch = WienerFilter(gauss_monarch);
psnr_wiener = PSNR(gray_monarch,wiener_monarch);

% 1.3. Constrained Least Square Restoration
const_monarch_1 = LeastSquareFilter(gauss_monarch,0.1);
const_monarch_2 = LeastSquareFilter(gauss_monarch,1.0);
const_monarch_3 = LeastSquareFilter(gauss_monarch,10);

psnr_const_1 = PSNR(gray_monarch,const_monarch_1);
psnr_const_2 = PSNR(gray_monarch,const_monarch_2);
psnr_const_3 = PSNR(gray_monarch,const_monarch_3);

psnr_const = psnr_const_2;

% 1.4. Draw the power spectra of original image and noise
orig_spectra = ShiftFreq(FFT(FFT(gray_monarch).').');
gauss_spectra = ShiftFreq(FFT(FFT(gauss_monarch).').');
p_inv_spectra = ShiftFreq(FFT(FFT(p_inv_monarch_3).').');
wiener_spectra = ShiftFreq(FFT(FFT(wiener_monarch).').');

%% Displaying figures
figure('Name', 'Problem 1.1. Pseudo-inverse');
subplot(1,4,1); imshow(uint8(gray_monarch)); title('Original Monarch');
subplot(1,4,2); imshow(uint8(abs(p_inv_monarch_1))); title('threshold = 0.3');
subplot(1,4,3); imshow(uint8(abs(p_inv_monarch_2))); title('threshold = 0.5');
subplot(1,4,4); imshow(uint8(abs(p_inv_monarch_3))); title('threshold = 0.7');

figure('Name','Problem 1.2. Wiener Filter');
subplot(1,2,1); imshow(uint8(gray_monarch)); title('Original Monarch');
subplot(1,2,2); imshow(uint8(abs(wiener_monarch))); title('Wiener Filtered Monarch');

figure('Name','Problem 1.3. Constrained Least Square Restoration');
subplot(1,4,1); imshow(uint8(gray_monarch)); title('Original Monarch');
subplot(1,4,2); imshow(uint8(abs(const_monarch_1))); title('\lambda = 0.1');
subplot(1,4,3); imshow(uint8(abs(const_monarch_2))); title('\lambda = 1.0');
subplot(1,4,4); imshow(uint8(abs(const_monarch_3))); title('\lambda = 10');

figure('Name','Problem 1.4. The power spectra');
subplot(1,4,1); imshow(uint8(20*log(abs(orig_spectra)))); title('Original Spectrum');
subplot(1,4,2); imshow(uint8(20*log(abs(gauss_spectra)))); title('Gaussian Noise Spectrum');
subplot(1,4,3); imshow(uint8(20*log(abs(p_inv_spectra)))); title('Pseudo-inverse Spectrum');
subplot(1,4,4); imshow(uint8(20*log(abs(wiener_spectra)))); title('Wiener Spectrum');

figure('Name','Problem 1.5. Compare and analyze the image characteristic from 1.1, 1.2 and 1.3');
subplot(1,4,1); imshow(uint8(gray_monarch)); title('Original Monarch');
subplot(1,4,2); imshow(uint8(abs(p_inv_monarch_2))); title('Pseudo-inverse, threshold = 0.5');
subplot(1,4,3); imshow(uint8(abs(wiener_monarch))); title('Wiener Filtered Monarch');
subplot(1,4,4); imshow(uint8(abs(const_monarch_2))); title('Constrained Least Square Restoration, \lambda = 1.0');

%% Print PSNR values
fprintf('Problem 1.1. Pseudo-inverse \n');
fprintf('threshold = 0.3: psnr_p_inverse_1 = %f dB \n', psnr_p_inverse_1);
fprintf('threshold = 0.5: psnr_p_inverse_2 = %f dB \n', psnr_p_inverse_2);
fprintf('threshold = 0.7: psnr_p_inverse_3 = %f dB \n', psnr_p_inverse_3);
fprintf('====\n');

fprintf('Problem 1.3. Constrained Least Square Restoration \n');
fprintf('lambda = 0.1: psnr_const_1 = %f dB \n', psnr_const_1);
fprintf('lambda = 1.0: psnr_const_2 = %f dB \n', psnr_const_2);
fprintf('lambda = 10: psnr_const_3 = %f dB \n', psnr_const_3);
fprintf('====\n');

fprintf('Problem 1.5. Compare and analyze the image characteristic from 1.1, 1.2 and 1.3 \n');
fprintf('threshold = 0.5: psnr_p_inverse = %f dB \n', psnr_p_inverse);
fprintf('psnr_wiener = %f dB \n', psnr_wiener);
fprintf('lambda = 1.0: psnr_const = %f dB \n', psnr_const);
fprintf('====\n');

%%---------------------------------------------------------------%%
end

% Add Gaussian noise function
function A = GaussianNoise(X,std)
	X = X / 255;
	[m,n] = size(X);
	u = rand(m,n);
	v = rand(m,n);
	GN = sqrt(-2*log(u)).*cos(2*pi*v);
	A = X + GN * std;
	A = A * 255;
end

% PSNR function
function psnr = PSNR(Orig,Dist)
	[m, n, p] = size(Orig);
	Orig = double(Orig);
	Dist = double(Dist);
	error = Orig - Dist;
	MSE = sum(sum(sum(error.^2)))/(m*n*p);
	if MSE > 0
    psnr = 20*log10(max(max(max(Orig)))) - 10*log10(MSE);
	else
    psnr = 99;
	end
end

% FFT Function
function A = FFT(X)
  N = length(X);

  X_even = X(:,1:2:N-1);
  X_odd = X(:,2:2:N);
  W = exp(-1j*4*pi/N);

  m = 0:1:(N/2-1);
  k = 0:1:(N-1);

  KM_even = m' * k;
  KM_odd = (m+0.5)' * k;

  WN_even = W.^KM_even;
  WN_odd = W.^KM_odd;

  A = X_even*WN_even + X_odd*WN_odd;
end

% Inverse FFT Function
function A = IFFT(X)
  N = length(X);

  X_even = X(:,1:2:N-1);
  X_odd = X(:,2:2:N);
  W = exp(1j*4*pi/N);

  m = 0:1:(N/2-1);
  k = 0:1:(N-1);

  KM_even = m' * k;
  KM_odd = (m+0.5)' * k;

  WN_even = W.^KM_even;
  WN_odd = W.^KM_odd;

  A = (X_even*WN_even + X_odd*WN_odd)/N;
end

% Hamming filter
function A = HammingFilter(X)
	[~,n] = size(X);
	t = 0:1:n-1;
	W1d = 0.54 - 0.46 * cos(2*pi*t/(n-1));
	W2d = W1d' * W1d;
	fftX = FFT(FFT(X).').';
	fftX = (ShiftFreq(fftX)).*W2d;
	A = IFFT(IFFT(ShiftFreq(fftX)).').';
end

function A = ShiftFreq(X)
	[m,n] = size(X);
	S1 = X(1:m/2, 1:n/2);
	S2 = X(1:m/2, n/2+1:n);
	S3 = X(m/2+1:m, 1:n/2);
	S4 = X(m/2+1:m, n/2+1:n);
	A = [S4 S3; S2 S1];
end

% Pseudo-inverse Filter
function A = PseudoInvFilter(X,threshold)
	[m,n] = size(X);
	fftX = FFT(FFT(X).').';		% Transform image into the frequency domain
	shiftX = ShiftFreq(fftX);	% Shift low frequency components to the middle image

	% Hamming window
	k = 0:1:n-1;
	W1d = 0.54 - 0.46 * cos(2*pi*k/(n-1));
	W2d = W1d' * W1d;

	% Pseudo-inverse mask
	mask = zeros(m,n);
	for i = 1:m
		for j = 1:n
			if abs(W2d(i,j)) < threshold
				mask(i,j) = 0;
			else
				mask(i,j) = 1/W2d(i,j);	
			end
		end
	end

	A = IFFT(IFFT(ShiftFreq(shiftX.*mask)).').';
end

% Wiener Filter
function A = WienerFilter(X)
	[~,n] = size(X);
	fftX = FFT(FFT(X).').';		% Transform image into the frequency domain
	shiftX = ShiftFreq(fftX);	% Shift low frequency components to the middle image
	snr = 10^1.4; 						% SNR = 14 dB

	% Hamming window
	k = 0:1:n-1;
	W1d = 0.54 - 0.46 * cos(2*pi*k/(n-1));
	W2d = W1d' * W1d;

	% Wiener mask
	mask = conj(W2d).*snr ./ (W2d.^2.*snr+1);

	A = IFFT(IFFT(ShiftFreq(shiftX.*mask)).').';
end

function A = LeastSquareFilter(X,lamda)
	[m,n] = size(X);

	fftX = FFT(FFT(X).').';		% Transform image into the frequency domain
	shiftX = ShiftFreq(fftX);	% Shift low frequency components to the middle image

	laplacian_operator = zeros(m,n);
	laplacian_operator(1:3,1:3) = [0 -1 0; -1 4 -1; 0 -1 0];
	Q = FFT(FFT(laplacian_operator).').';
	Q = ShiftFreq(Q);

	% Hamming window
	k = 0:1:n-1;
	W1d = 0.54 - 0.46 * cos(2*pi*k/(n-1));
	W2d = W1d' * W1d;

	% Least Squared mask
	mask = conj(W2d) ./ (W2d.^2+lamda*abs(Q).^2);

	A = IFFT(IFFT(ShiftFreq(shiftX.*mask)).').';
end
