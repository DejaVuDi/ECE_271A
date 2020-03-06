clc;
close all;
clear all;
load('TrainingSamplesDCT_8_new.mat'); % training data
zig_python = load('Zig-Zag Pattern.txt'); % pattern
zig = zig_python+1; % for matlab
zz=load('Zig-Zag Pattern.txt');
PC = (250/(250+1053)); % MLE
PG = 1-PC;

ind = [1,8,12,24,25,26,33,40];
good_FG = TrainsampleDCT_FG(:,ind);
good_BG = TrainsampleDCT_BG(:,ind) ;

mu_c = mean(good_FG);
mu_g = mean(good_BG); % mu

sigma_c = cov(good_FG);
sigma_g = cov(good_BG);
% Pxc = (1/sqrt((2*pi^64)*abs(sigma_c)))*exp(-1/2*(x-mu_c)*(x-mu_c))

ch = im2double(imread('cheetah.bmp')); % make it in double
padded = padarray(ch,[7,7],'symmetric','post'); % padded image
[m, n] = size(padded);
for i1 = 1:(m-7)
    for j1 = 1:(n-7)
        % compute position of 2nd largest magnitude
        block = padded(i1:i1+7, j1:j1+7);
        dct_value = dct2(block);
        for col = 1:64
            [v1, v2] = find(zig == col);
            v3(col) = dct_value(v1, v2);
        end
        dct_sort = v3(:,ind);
        % compute A
        if log(mvnpdf(dct_sort,mu_c,sigma_c))+log(PC) > log(mvnpdf(dct_sort,mu_g,sigma_g))+log(PG)
            A(i1,j1) = 1;
        else
            A(i1,j1) = 0;
        end
    end
end

figure();
imagesc(A) % plot image
colormap(gray(255))
[f, g] = size(A);
mask = im2double(imread('cheetah_mask.bmp'));
diff = mask - A;
mask_ch = sum((sum(mask==1)),2);
E_ch = sum((sum(diff==1)),2);
mask_g = sum((sum(mask==0)),2);
E_g = sum((sum(diff==-1)),2);
PoE = (E_ch/mask_ch)*PC+(E_g/mask_g)*PG;