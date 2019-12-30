clear;

Problem_2();

function Problem_2()
% Template for EE535 Digial Image Processing
% Insert the code in the designated area below
%% Loading directory for image files
imgdir = uigetdir('D:\KAIST\Courses\dip\hw\hw5\Test_images');

file = fopen(fullfile(imgdir,'\texture1_gray_256x256.raw'),'rb');
gray_texture1 = fread(file,fliplr([256,256]),'*uint8')';
fclose(file);

file = fopen(fullfile(imgdir,'\texture2_gray_256x256.raw'),'rb');
gray_texture2 = fread(file,fliplr([256,256]),'*uint8')';
fclose(file);

file = fopen(fullfile(imgdir,'\object_gray_360x285.raw'),'rb');
gray_object = fread(file,fliplr([360,285]),'*uint8')';
fclose(file);

file = fopen(fullfile(imgdir,'\hough_gray_256x256.raw'),'rb');
gray_hough = fread(file,fliplr([256,256]),'*uint8')';
fclose(file);
%%
%%---------------------Insert code below ----------------------%%
%% 2. Image Analysis
gray_texture1 = double(gray_texture1);
gray_texture2 = double(gray_texture2);
gray_object = double(gray_object);
gray_hough = double(gray_hough);

% 2.1. Transform features
[t1_45_img, t1_45] = AngularSlits(gray_texture1,45);
[t1_135_img, t1_135] = AngularSlits(gray_texture1,135);
[t2_45_img, t2_45] = AngularSlits(gray_texture2,45);
[t2_135_img, t2_135] = AngularSlits(gray_texture2,135);

fft_t1 = FFT(FFT(gray_texture1).').';
shift_t1 = ShiftFreq(fft_t1);
freq_t1 = 10*log(abs(shift_t1));

fft_t1_45 = FFT(FFT(t1_45_img).').';
shift_t1_45 = ShiftFreq(fft_t1_45);
freq_t1_45 = 10*log(abs(shift_t1_45));

fft_t1_135 = FFT(FFT(t1_135_img).').';
shift_t1_135 = ShiftFreq(fft_t1_135);
freq_t1_135 = 10*log(abs(shift_t1_135));

fft_t2 = FFT(FFT(gray_texture2).').';
shift_t2 = ShiftFreq(fft_t2);
freq_t2 = 10*log(abs(shift_t2));

fft_t2_45 = FFT(FFT(t2_45_img).').';
shift_t2_45 = ShiftFreq(fft_t2_45);
freq_t2_45 = 10*log(abs(shift_t2_45));

fft_t2_135 = FFT(FFT(t2_135_img).').';
shift_t2_135 = ShiftFreq(fft_t2_135);
freq_t2_135 = 10*log(abs(shift_t2_135));

% 2.2. Threshold-based Segmentation
threshold = Otsu(gray_object);
extracted_obj = ThreshSegm(gray_object,threshold);

% 2.3. General Hough Transform
[detected_lines,hough_parameter_space] = HoughTransform(gray_hough);

%% Displaying figures
figure('Name','Problem 2.1. Transform Features');
subplot(3,4,1); imshow(uint8(abs(gray_texture1))); title('texture 1 in spatial domain');
subplot(3,4,2); imshow(uint8(freq_t1)); title('texture 1 in frequency domain');
subplot(3,4,3); imshow(uint8(freq_t1_45)); title('texture 1 applied 45 slit in frequency domain');
subplot(3,4,4); imshow(uint8(abs(t1_45_img))); title('Inverse texture 1 after applying 45 slit in spatial domain');
subplot(3,4,5); imshow(uint8(freq_t1_135)); title('texture 1 applied 135 slit in frequency domain');
subplot(3,4,6); imshow(uint8(abs(t1_135_img))); title('Inverse texture 1 after applying 135 slit in spatial domain');
subplot(3,4,7); imshow(uint8(abs(gray_texture2))); title('texture 2 in spatial domain');
subplot(3,4,8); imshow(uint8(freq_t2)); title('texture 2 in frequency domain');
subplot(3,4,9); imshow(uint8(freq_t2_45)); title('texture 2 applied 45 slit in frequency domain');
subplot(3,4,10); imshow(uint8(abs(t2_45_img))); title('Inverse texture 2 after applying 45 slit in spatial domain');
subplot(3,4,11); imshow(uint8(freq_t2_135)); title('texture 2 applied 135 slit in frequency domain');
subplot(3,4,12); imshow(uint8(abs(t2_135_img))); title('Inverse texture 2 after applying 135 slit in spatial domain');

figure('Name','Problem 2.2. Threshold-based Segmentation');
subplot(1,2,1); imshow(uint8(gray_object)); title('Original Gray Object');
subplot(1,2,2); imshow(uint8(extracted_obj)); title('Etracted Object Image');

figure('Name','Problem 2.3. General Hough Transform');
subplot(1,3,1); imshow(uint8(abs(gray_hough))); title('Input image');
subplot(1,3,2); imshow(uint8(abs(hough_parameter_space))); title('Parameter space image');
subplot(1,3,3); imshow(uint8(abs(detected_lines))); title('Detected straight-lines image');	

%% Print energy ratio
fprintf('Problem 2.1. Transform Features \n');
fprintf('t1_45 = %f \n', t1_45);
fprintf('t1_135 = %f \n', t1_135);
fprintf('t2_45 = %f \n', t2_45);
fprintf('t2_135 = %f \n', t2_135);
fprintf('====\n');

fprintf('Problem 2.2. Threshold-based Segmentation \n');
fprintf('threshold = %f \n', threshold);
fprintf('====\n');

%%---------------------------------------------------------------%%
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

function A = ShiftFreq(X)
	[m,n] = size(X);
	S1 = X(1:m/2, 1:n/2);
	S2 = X(1:m/2, n/2+1:n);
	S3 = X(m/2+1:m, 1:n/2);
	S4 = X(m/2+1:m, n/2+1:n);
	A = [S4 S3; S2 S1];
end

% Angular Slits
function [A,energy_ratio] = AngularSlits(X,Ang)
	[m,~] = size(X);
	mask = zeros(m,m);

	if Ang == 45
		for i = 50:200
			for j = 50:200
				mask(i,j) = (j < (262 - i)) && (j > (250 - i));
			end
		end
	end

	if Ang == 135
		for i = 50:200
			for j = 50:200
				mask(i,j) = (j < (i + 6)) && (j > (i - 6));
			end
		end 
	end

	fftX = FFT(FFT(X).').';
	shiftX = ShiftFreq(fftX);
	A = IFFT(IFFT(ShiftFreq(shiftX.*mask)).').';
	energy_ratio = (sum(sum(abs(shiftX.*mask)).^2) / sum(sum(abs(shiftX).^2)));
end

% Histogram Equalization
function A = histEq(X)
	[m, n] = size(X);

	% Get pdf
	P = zeros(256,1);
	for i = 1:256
		P(i) = sum(sum(X==i-1))/(m*n);
	end

	% Get cdf
	C = zeros(256,1);
	C(1) = P(1);
	for i = 2:256
		C(i) = C(i-1) + P(i);
	end

	% Mapping
	T = C * 255;
	A = zeros(m,n);
	for i = 1:m
		for j = 1:n
			A(i,j) = T(uint8(X(i,j))+1);
		end
	end
end

% Compass Operators
function [A,D] = CompassOperators(X)
	% A: ouput image
	% D: direction
	[m,n] = size(X);
	D = zeros(m,n);
	Ksize = 3;

	% Zero-padding
	Z = zeros(m+Ksize-1,n+Ksize-1);
	p = floor(Ksize/2);
	r = Ksize - 1;
	for i = 1:m
		for j = 1:n
			Z(i+p,j+p) = X(i,j);
		end
	end

	A = ones(m,n);

	% Gradient
	for i = 1:m
		for j = 1:n
			% 8 directions: N, NW, W, SW, S, SE, E, NE
			V = zeros(1,8);
			Ang = [pi/2,3*pi/4,pi,5*pi/4,3*pi/2,7*pi/4,2*pi,9*pi/4];

			% North (N)
			N = [1 1 1; 0 0 0; -1 -1 -1];
			count = 1;
			V(count) = sum(sum(N.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% North-West (NW)
			NW = [1 1 0; 1 0 -1; 0 -1 -1];
			V(count) = sum(sum(NW.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% West (W)
			W = [1 0 -1; 1 0 -1; 1 0 -1];
			V(count) = sum(sum(W.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% South-West (SW)
			SW = [0 -1 -1; 1 0 -1; 1 1 0];
			V(count) = sum(sum(SW.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% South (S)
			S = [-1 -1 -1; 0 0 0; 1 1 1];
			V(count) = sum(sum(S.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% South-East (SE)
			SE = [-1 -1 0; -1 0 1; 0 1 1];
			V(count) = sum(sum(SE.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% East (E)
			E = [-1 0 1; -1 0 1; -1 0 1];
			V(count) = sum(sum(E.*Z(i:i+r,j:j+r)));
			count = count + 1;

			% North-East (NE)
			NE = [0 1 1; -1 0 1; -1 -1 0];
			V(count) = sum(sum(NE.*Z(i:i+r,j:j+r)));

			[val, idx] = max(abs(V));
			D(i,j) = (Ang(idx)-pi/2)/pi;
			A(i,j) = val; % Compass gradient
		end
	end
end

function [A,parameter_space_img] = HoughTransform(X)
	[m,n] = size(X);
	y_max = round(sqrt(m^2+n^2));
	t_max = 360;
	[compass_img,~] = CompassOperators(X);
	for i = 1:m
		for j = 1:n
			compass_img(i,j) = (compass_img(i,j)>30);
		end
	end

	parameter_space = zeros(t_max,y_max);
	for i = 1:m
		for j = 1:n
			if compass_img(i,j) == 1
				for k = 1:t_max
					r = round(i*cos(k*pi/180)+j*sin(k*pi/180));
					if ((r > 0) && (r < (y_max + 1)))
						parameter_space(k,r) = parameter_space(k,r) + 1;
					end
				end
			end
		end
	end

	parameter_space_img = histEq(parameter_space);

	% Draw lines
	[row,col] = find(parameter_space > 60);
	for i = 1:size(row)
		lineEq = -1/tan(row(i)*pi/180);
		c = col(i) / sin(row(i)*pi/180);
		for j = 1:m
			L = round(lineEq*j+c);
			if (L>0 && L<n+1)
				X(j,L) = 255;
			end
			if (row(i)>357 || row(i)<3)
				X(col(i),:) = 255;
			end
		end
	end
	A = X;
end

function threshold = Otsu(X)
    [m,n] = size(X);
    sizeX = m * n;
    maximum = 0;
    
    % Get pdf
	P = zeros(1,256);
	for i = 1:256
		P(i) = sum(sum(X==i-1))/sizeX;
    end
    
    % Step through intensity level from 2 to 255
    for k = 2:255
       w0 = sum(P(1:k)); % Probability of class 0
       w1 = sum(P(k+1:256)); % Probability of class 1
       u0 = sum((0:k-1).*P(1:k)) / w0; % Mean of class 0
       u1 = sum((k:255).*P(k+1:256)) / w1; % Mean of class 1
       sigma = w0 * w1 * (u1-u0)^2;
       if sigma > maximum
          maximum = sigma;
          threshold = k - 1;
       end
    end
end

function A = ThreshSegm(X,threshold)
    [m,n] = size(X);
    A = zeros(m,n);
    for i = 1:m
       for j = 1:n
          if X(i,j) > threshold
             A(i,j) = 255; 
          end
       end
    end
end

