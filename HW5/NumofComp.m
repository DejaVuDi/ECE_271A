%% Initialization
clear;clc;
tic;
 
load('TrainingSamplesDCT_8_new.mat'); % training data
ch = im2double(imread('cheetah.bmp')); % make it in double
zig = load('Zig-Zag Pattern.txt');
d = 64; % dimensions
C_ = [1,2,4,8,16,32]; % components
mixture = 5; % mixtures
iter = 10; % iterations
e = 1e-4;
 
TS_c = TrainsampleDCT_FG(:,1:d);
TS_g = TrainsampleDCT_BG(:,1:d);
nc = length(TrainsampleDCT_FG);
ng = length(TrainsampleDCT_BG);
PC = nc/(nc+ng); % foreground
PG = 1-PC; % background
[m,n] = size(ch);
padded = padarray(ch,[7,7],'symmetric','post'); % padded image
 
% dct all
test = zeros(m*n,d); 
for i = 1:m
    for j = 1:n
        block = padded(i:i+7,j:j+7);
        dct_value = dct2(block);
        temp = zeros(1,64);
        for x = 1:8
            for y = 1:8
                temp(zig(x,y)+1) = dct_value(x,y);
            end
        end
        test((i-1)*n+j,:) = temp;
    end
end
 
paramix_c = cell(length(C_),2);
paramix_g = cell(length(C_),2);
 
for mix = 1:length(C_)
    % parameters for FG
    C = C_(mix);     
    param2 = cell(C,2);            
    for i = 1:C                     
        param2{i,1} = randn(d,1);
        param2{i,2} = diag(3+rand(d,1));
    end        
    h_c = zeros(nc,C);        
    nh_c = 0;        
    Pi_c = ones(1,C)/C;        
    px_c = zeros(nc,C);        
    for it = 1:iter          
        for k = 1:C
            px_c(:,k) = mvnpdf(TS_c,param2{k,1}',param2{k,2});
        end
        for i = 1:C
            % E
            h_c(:,i) = mvnpdf(TS_c,param2{i,1}',param2{i,2})*Pi_c(i); 
            h_c(:,i) = h_c(:,i)./(px_c*Pi_c');  
            % M
            nh_c = sum(h_c(:,i));       
            Pi_c(i) = nh_c/size(h_c,1); 
            param2{i,1} = sum(repmat(h_c(:,i),1,d).*TS_c)'/nh_c; 
            sigma_update = zeros(d,d);
            for k = 1:size(h_c,1)
                xi_muj = TS_c(k,:)-param2{i,1}';
                sigma_update = sigma_update+h_c(k,i)*diag(diag(xi_muj'*xi_muj));
            end
            param2{i,2} = sigma_update/nh_c+e; 
        end
    end
    paramix_c{mix,1} = param2;
    paramix_c{mix,2} = Pi_c;
    % parameters for BG
    param1 = cell(C,2);
    for c = 1:C
        param1{c,1} = randn(d,1);
        param1{c,2} = diag(6+rand(d,1).^2);
    end
    Pi_g = ones(1,C)/C;
    h_g = zeros(ng,C);
    nh_g = 0;
    px_g = zeros(ng,C);
    for it_no = 1:iter    
        for k = 1:C
            px_g(:,k) = mvnpdf(TS_g,param1{k,1}',param1{k,2});
        end
        for i = 1:C          
            % E            
            h_g(:,i) = mvnpdf(TS_g,param1{i,1}',param1{i,2})*Pi_g(i);             
            h_g(:,i) = h_g(:,i)./(px_g*Pi_g');               
            % M
            nh_g = sum(h_g(:,i));       
            Pi_g(i) = nh_g/size(h_g,1);   
            param1{i,1} = sum(repmat(h_g(:,i),1,d).*TS_g)'/nh_g;          
            sigma_update = zeros(d,d);            
            for k = 1:ng
                xi_muj = TS_g(k,:)-param1{i,1}';
                sigma_update = sigma_update+h_g(k,i)*diag(diag(xi_muj'*xi_muj));
            end            
            param1{i,2} = sigma_update/nh_g+e;   
        end
    end
    paramix_g{mix,1} = param1;
    paramix_g{mix,2} = Pi_g; 
end
%% Classification
dim = [1,2,4,8,16,24,32,40,48,56,64];
PoE = zeros(length(dim),length(C_));
% PoE = zeros(1,length(d),5,5);
for dims = 1:length(dim)
    dims
    di = dim(dims);
    for a = 1:length(C_)
        C = C_(a);
        p_x_g = 0;
        p_x_c = 0;
        for no = 1:C
            p_x_g = p_x_g+mvnpdf(test(:,1:di),paramix_g{a,1}{no,1}(1:di)',paramix_g{a,1}{no,2}(1:di,1:di))*paramix_g{a,2}(no);
            p_x_c = p_x_c+mvnpdf(test(:,1:di),paramix_c{a,1}{no,1}(1:di)',paramix_c{a,1}{no,2}(1:di,1:di))*paramix_c{a,2}(no);
        end
        A = zeros(size(ch)); 
        for row = 1:m 
            for col = 1:n
                zi = (row-1)*n+col;
                if col < n-7
                    if p_x_g(zi)*PG > p_x_c(zi)*PC
                       A(row,col) = 0;
                    else
                        A(row,col) = 1;
                    end      
                end
            end
        end
        PoE(dims,a) = error(A,PC,PG);
    end
end
figure(1)
imagesc(A);
toc;
%% plot
figure
mk = ['o','+','*','s','d','x'];
for i = 1:length(C_)
 
    plot(dim,PoE(:,i),'Marker',mk(i),'LineWidth',.8);
    hold on
end
title('PoE for different # of mixture components');
xlabel('Dimension');
ylabel('PoE');
legend('C=1','C=2','C=4','C=8','C=16','C=32','Location','northwest');
xlim auto;
ylim auto;
saveas(gcf,strcat('b.autosave.bmp'));
%% function
function pe = error(A,PC,PG)
    m = 250;
    n = 270;
    mask = im2double(imread('cheetah_mask.bmp'));
    A1 = A(1:m-7,1:n-7);
    A1 = padarray(A1,[4,4],'pre');
    A1 = padarray(A1,[3,3],'post');
    a1 = 0;b1 = 0;
    a2 = 0;b2 = 0;
    for row = 1:m
        for col = 1:n                
            if mask(row,col)==1&&A1(row,col)==1
                b1 = b1+1;
                a1 = a1+1;
            elseif mask(row,col)==1
                b1 = b1+1;
            end                    
            if mask(row,col)==0&&A1(row,col)==1
                b2 = b2+1;
                a2 = a2+1;
            elseif mask(row,col)==0
                b2 = b2+1;
            end   
        end
    end
    ecc = a1/b1;ecg = a2/b2;
    pe = ecg*PG+(1-ecc)*PC; 
end
