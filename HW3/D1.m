%% load value
clc;
close all;
clear all;
global W0 zig PC PG alpha mu_c mu_g sigma_c sigma_g;
load('TrainingSamplesDCT_subsets_8.mat'); % training data
load('Alpha.mat');
load('Prior_1.mat');
zig_python = load('Zig-Zag Pattern.txt'); % pattern
zig = zig_python+1; % for matlab
ch = im2double(imread('cheetah.bmp')); % make it in double
mask = im2double(imread('cheetah_mask.bmp'));
d = 64;

nc = length(D1_FG);
ng = length(D1_BG);
PC = nc/(nc+ng); % foreground
PG = 1-PC; % background

% P_x_i = @(x,sigma_i,mean_i,d)(1/(sqrt((2*pi)^d*det(sigma_i)))*exp(-0.5*(x-mean_i)'*(sigma_i\(x-mean_i))));
classify = @(P_FG,PC,P_BG,PG)((log(P_FG)+log(PC) > log(P_BG)+log(PG))*1);
% find sigma_i and mean_i
mu_i = @(sigma0,n,sigma,mu,mu0)((sigma0*((sigma0+(1/n)*sigma)\mu'))+((1/n)*sigma*((sigma0+(1/n)*sigma)\mu0')));
sigma_i = @(sigma0,n,sigma)(sigma0*((sigma0+sigma/n)\sigma)/n); 

mu_c = mean(D1_FG); % mu_hat_n (sample mean)
mu_g = mean(D1_BG);
sigma_c = cov(D1_FG); % sigma_n (sample covariance) 
sigma_g = cov(D1_BG);
%% plot
mu_i_c = @(sigma0)mu_i(sigma0,nc,sigma_c,mu_c,mu0_FG);
mu_i_g = @(sigma0)mu_i(sigma0,ng,sigma_g,mu_g,mu0_BG);
sigma_i_c = @(sigma0)sigma_i(sigma0,nc,sigma_c);
sigma_i_g = @(sigma0)sigma_i(sigma0,ng,sigma_g);

P_bpe_FG = @(x,sigma0)mvnpdf(x,mu_i_c(sigma0),(sigma_c+sigma_i_g(sigma0)));
P_bpe_BG = @(x,sigma0)mvnpdf(x,mu_i_g(sigma0),(sigma_g+sigma_i_g(sigma0)));
P_ml_FG = @(x)mvnpdf(x,mu_c',sigma_c);
P_ml_BG = @(x)mvnpdf(x,mu_g',sigma_g);
P_map_FG = @(x,sigma0)mvnpdf(x,mu_i_c(sigma0),sigma_i_g(sigma0));
P_map_BG = @(x,sigma0)mvnpdf(x,mu_i_g(sigma0),sigma_i_g(sigma0));

[A_bpe,PoE_bpe] = map(ch,@(x,sigma0)(classify(P_bpe_FG(x,sigma0),PC,P_bpe_BG(x,sigma0),PG)),1,mask);
[A_ml,PoE_ml] = map(ch,@(x)(classify(P_ml_FG(x),PC,P_ml_BG(x),PG)),2,mask);
[A_map,PoE_map] = map(ch,@(x,sigma0)(classify(P_map_FG(x,sigma0),PC,P_map_BG(x,sigma0),PG)),1,mask);

plot(alpha,PoE_bpe,'b--o',alpha,PoE_ml,'--',alpha,PoE_map,'-*')
legend('BPE','ML','MAP');
xlabel('\alpha');
ylabel('PoE');
xlim auto;
ylim auto;
set(gca,'XScale','log');
title('\alpha vs PoE on D1')

%% function
function [A,PoE] = map(ch,classify,flag,mask)
    global W0 zig PC PG alpha;
    padded = padarray(ch,[7,7],'symmetric','post'); % padded image
    [m, n] = size(padded);
    A = zeros(255,270);
    for a = 1:9
        sigma0 = diag(alpha(a)*W0);
        for i1 = 1:(m-7)
            for j1 = 1:(n-7)
                block = padded(i1:i1+7, j1:j1+7);
                dct_value=dct2(block);
                temp=zeros(64,1);
                for x=1:8
                    for y=1:8
                        temp(zig(x,y))=dct_value(x,y);
                    end
                end
                if j1<(n-14)
                    if flag == 1
                        A(i1,j1) = classify(temp,sigma0);
                    elseif flag == 2
                        A(i1,j1) = classify(temp);
                    end
                end
            end
        end 
    diff = mask - A;
    mask_ch = sum((sum(mask==1)),2);
    E_ch = sum((sum(diff==1)),2);
    mask_g = sum((sum(mask==0)),2);
    E_g = sum((sum(diff==-1)),2);
    PoE(a) = (E_ch/mask_ch)*PC+(E_g/mask_g)*PG;
    end
end