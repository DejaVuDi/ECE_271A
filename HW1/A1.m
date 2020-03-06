clc;
close all;
clear all;
load('TrainingSamplesDCT_8.mat'); % training data
zig_python = load('Zig-Zag Pattern.txt'); % pattern
zig = zig_python+1; % for matlab
PC = (250/(250+1053)); % prior use training sample
PG = 1-PC;
TC = abs(TrainsampleDCT_FG); % indexing
TG = abs(TrainsampleDCT_BG);
x = sort(TC,2,'descend');
y = sort(TG,2,'descend');
% histograms
 for i = 1:250
     a(i) = find(TC(i,:)==x(i,2));
 end
 for j = 1:1053
     b(j) = find(TG(j,:)==y(j,2));
 end
histogram(a,'Normalization','pdf');
figure();
histogram(b,'Normalization','pdf');
Pxc = histcounts(a, 1:64);
Pxc = Pxc/250;
Pxg = histcounts(b, 1:64);
Pxg = Pxg/1053;
ch = im2double(imread('cheetah.bmp')); % make it in double
padded = padarray(ch,[1,2],'post'); % padded image for 256*272
[m, n] = size(padded);
for i1 = 1:(m-7)
    for j1 = 1:(n-7)
        block = padded(i1:i1+7, j1:j1+7);
        dct_value = abs(dct2(block));
        for col = 1:64
            [v1, v2] = find(zig==col);
            v3(col) = dct_value(v1, v2);
        end
        dct_sort = sort(v3, 'Descend');
        mag = dct_sort(2);
        position = find(v3==mag);
        feature(i1,j1) = position;
        if Pxc(feature(i1,j1))*PC > Pxg(feature(i1,j1))*PG
            A(i1,j1) = 1;
        else
            A(i1,j1) = 0;
        end
    end
end
figure();
imagesc(A)
colormap(gray(255))
mask = im2double(imread('cheetah_mask.bmp'));
error = abs(mask(1:249,1:265)- A);
sm = sum(error);
PofE = sum(sm,2)/(m*n);    