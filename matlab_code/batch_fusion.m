
close all
clear all
clc
tic
sample = num2str(260513);
psf_dir1 = double(imread('your/file/path/PSF100_1_x.bmp')); 
psf_dir2 = double(imread('your/file/path/PSF100_1_y.bmp'));
psf_dir1 = psf_dir1 / max(psf_dir1(:));
psf_dir2 = psf_dir2 / max(psf_dir2(:));


% fpath = strcat('your/file/path/input/');
for ch = 1
    for layer = 1

%         fpath1 = strcat(sample, '_CH', num2str(ch), '_', num2str(layer, '%05d'), '\'); 
% %         fpath2 = strcat(sample, '_CH', num2str(ch), '_', num2str(layer, '%05d'), '\');
%         fpath2 = strcat('your/file/path/input/');
% 
%         fpath_x_raw = strcat(fpath, 'CH', num2str(ch), '\', fpath1);  
% %         fpath_y_raw = strcat(fpath, 'input/', 'CH', num2str(ch), '\', fpath2);
%         fpath_y_raw = strcat(fpath, fpath2); 
% %         fpath_deconv = strcat(fpath, 'output/', 'CH', num2str(ch), '\', fpath2); 
%         fpath_deconv = strcat(fpath, 'your/file/path/output/');

        fpath_x_raw = strcat('your/file/path/input_x/');
        fpath_y_raw = strcat('your/file/path/input_y/');
        fpath_deconv = strcat('your/file/path/output/');
        
        if ~exist(fpath_deconv, 'dir')
            mkdir(fpath_deconv);
        end
        
        for strip = 40: 47 
%             fname = strcat('VCLS_', num2str(layer, '%05d'), '_', num2str(strip), '_CH', num2str(ch), '.tif'); 
%             fname1 = strcat(sample, '_', num2str(layer, '%05d'), '_', num2str(strip), '_CH', num2str(ch), '_SD1_3', '.tif'); % x
%             fname2 = strcat('yLC_', num2str(layer, '%05d'), '_', num2str(strip), '_CH', num2str(ch), '.tif');  % y

            fname = strcat('SCALE_', num2str(layer, '%05d'), '_', num2str(strip),'_CH', num2str(ch), '.tif'); 
            fname1 = strcat('line_', num2str(layer, '%05d'), '_', num2str(strip), '_CH', num2str(ch), '.tif'); % x
            fname2 = strcat('y_lc_', num2str(layer, '%05d'), '_', num2str(strip), '_CH', num2str(ch),  '.tif');  % y

            
            fname_deconv = strcat(fpath_deconv, fname);
            fname_x_raw = strcat(fpath_x_raw, fname1);
            fname_y_raw = strcat(fpath_y_raw, fname2);


            x_raw = double(imread(fname_x_raw));
            [m, n] = size(x_raw);
            y_raw = double(imread(fname_y_raw));
%             x_raw = x_raw / max(x_raw(:)) * max(y_raw(:));
            y_raw = y_raw / max(y_raw(:)) * max(x_raw(:));
            disp(strcat(datestr(now), ': data is read'));


            recon = dual_view_deconv_sparse(x_raw, y_raw, psf_dir1, psf_dir2, 3);
            disp(strcat(datestr(now), ': data is deconved'));


            imwrite(uint16(recon), fname_deconv);
            disp(strcat(datestr(now), ': data is written'));
        end
    end
end
