%%
close all
clear all
clc
tic
%%

fname_CH1_raw = 'your/file/path/2.tif';
% fname_CH1_ali = 'your/file/path/6.tif';
fname_deconv = 'your/file/path/0.tif';

%% 
CH1_raw = double(imread(fname_CH1_raw));
[m,n]=size(CH1_raw);
CH2_raw = double(imread(fname_CH2_raw));
CH2_raw = CH2_raw./max(CH2_raw(:))*max(CH1_raw(:));
% CH1_raw = CH1_raw./max(CH1_raw(:))*max(CH2_raw(:));
disp(strcat(datestr(now),':data is read'));

% %tic
% % CH1_ali =  corr_2D_redundance_iter(CH1_raw,CH2_raw,5,3)
% CH2_ali =  corr_2D_redundance_iter(CH2_raw,CH1_raw,5,3);
% %toc
% % –¥»Î
% % imwrite(uint16(CH2_ali),fname_CH1_ali);
% disp(strcat(datestr(now),':data is registered'));

%% 
psf_dir1 = double(imread('your/file/path/dcLiMo_x.bmp'));
psf_dir2 = double(imread('your/file/path/dcLiMo_y.bmp'));
psf_dir1 = psf_dir1/max(psf_dir1(:));
psf_dir2 = psf_dir2/max(psf_dir2(:));
recon = dual_view_deconv_sparse(CH1_raw,CH2_raw,psf_dir1,psf_dir2,3);
disp(strcat(datestr(now),':data is deconved'));

%% 

imwrite(uint16(recon),fname_deconv);
disp(strcat(datestr(now),':data is written'));
