close all
clear all
clc
img = double(imread('your/file/path/LiMo_00001_40_CH1.bmp'));
[rows, cols] = size(img);


center_width = round(10);
center_height = round(10);
x_start = round((cols - center_width) / 2);
y_start = round((rows - center_height) / 2);

mask = zeros(rows, cols);
mask(y_start:y_start + center_height - 1, x_start:x_start + center_width - 1) = 1;


modified_img = img .* mask;
symmetric_img = modified_img;
for i = 1:rows
    for j = 1:cols
        if j < i
            
            symmetric_img(i,j) = modified_img(j,i);
        end
    end
end
img_sym2 = symmetric_img;


for i = 1:rows
    for j = 1:cols
        if i + j > rows + 1 
           
            img_sym2(i,j) = symmetric_img(rows-j+1, cols-i+1);
        end
    end
end


imshow(img_sym2, []); 
imwrite(uint8(img_sym2), 'your/file/path/output_symmetric_image.bmp'); 