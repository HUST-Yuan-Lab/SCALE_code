close all
clear all
clc

input_folder = 'your/file/path/input/'; 
output_folder = 'your/file/path/output/'; 

if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end


channels = 2;  
layers = 103;    
strips = (29: 35); 

for ch = channels
    for layer = layers
        for strip = strips
            
            %input_folder_ch_layer = sprintf('%sCH%d/sample_CH%d_%05d/', input_folder, ch, ch, layer); 
            input_folder_ch_layer = input_folder;

            input_filename = sprintf('%slimoce_%05d_%02d_CH%d.tif', input_folder_ch_layer, layer, strip, ch);
            
            if exist(input_filename, 'file')

                info = imfinfo(input_filename);
                num_images = numel(info);
                
                if num_images == 2 
                    img1 = imread(input_filename, 1);  % limoc£¨TDI2£©
                    img2 = imread(input_filename, 2);  % limoe£¨TDI1£©
                    
                    if info(1).BitDepth ~= 16
                        warning('Image is not 16-bit: %s', input_filename);
                    end
                    
           
                    img1_enhanced = min(img1 * 4, 65535);  
                    img2_enhanced = min(img2 * 4, 65535);  
                    
           
                    result_img = img1_enhanced - img2_enhanced;
                    
                    result_img = uint16(result_img)-750; 
                    

                    %output_filename = sprintf('%sCH%d/sample_CH%d_%05d/y_limo_%05d_%02d_CH%d.tif', output_folder, ch, ch, layer, layer, strip, ch);
                    output_filename = sprintf('%sy_l1_limo_%05d_%02d_CH%d.tif', output_folder, layer, strip, ch);
                    
                    [output_dir, ~, ~] = fileparts(output_filename);
                    if ~exist(output_dir, 'dir')
                        mkdir(output_dir);
                    end
                    

                    imwrite(result_img, output_filename, 'TIFF');
                    fprintf('Processed and saved: %s\n', output_filename);
                else
                    warning('Image does not have exactly two layers: %s', input_filename);
                end
            else
                warning('File does not exist: %s', input_filename);
            end
        end
    end
end
