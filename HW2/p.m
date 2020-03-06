clc;
close all;
clear all;
load('TrainingSamplesDCT_8_new.mat'); % training data
figure
for i = 1:64
    subplot(8,8,i)
    FG = TrainsampleDCT_FG(:,i);
    BG = TrainsampleDCT_BG(:,i);
    [mu_f,s_f,muci_f,sci_f] = normfit(FG);
    [mu_b,s_b,muci_b,sci_b] = normfit(BG);
    f_f = linspace(mu_f-3*s_f,mu_f+3*s_f,100);
    f_b = linspace(mu_b-3*s_b,mu_b+3*s_b,100);
    norm_f = normpdf(f_f,mu_f,s_f);
    norm_b = normpdf(f_b,mu_b,s_b);
    plot(f_f,norm_f,f_b,norm_b)
    title(sprintf('feature %d',i))
end

%mu_F = mean(TrainsampleDCT_FG(:,1));
%sigma_F = var(TrainsampleDCT_FG(:,1));
%mu_B = mean(TrainsapleDCT_BG(:,1));
%sigma_B = var(TrainsampleDCT_BG(:,1));
%s = 
%f1 = normpdf(s,mu_F,sigma_F);
%f2 = normpdf(s,mu_B,sigma_B);
%figure;