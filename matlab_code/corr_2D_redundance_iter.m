function [tt,shift_I1,max_corr,shift_x,shift_y] = corr_2D_redundance_iter(CH1,CH2,rongyu_hh,iter_times)


I1 = CH1;
I2 = CH2;
thresh_signal = max(mean(I1(:)),mean(I2(:)));
thresh_corr = 0.95;
%thresh_corr_diff = 0.02;
rongyu_h = rongyu_hh*2;
[m,n] = size(I1);

%I1 = max(CH1-thresh_signal,0);
%I2 = max(CH2-thresh_signal,0);

for iter = 1:iter_times
    %clear tt
    if iter == 1
        step = n;
        total_ = 0;
        max_corr = zeros(ceil(m/step),ceil(n/step));
        shift_x = zeros(ceil(m/step),ceil(n/step));
        shift_y = zeros(ceil(m/step),ceil(n/step));
    else
        step = round(step/2);
        I1 = shift_I1;
        I2 = I2(rongyu_hh+1:m+rongyu_hh,rongyu_hh+1:n+rongyu_hh);
        max_corr = imresize(max_corr,2,'nearest');
        %shift_x = zeros(size(max_corr,1),size(max_corr,2));
        %shift_y = zeros(size(max_corr,1),size(max_corr,2));
        shift_x = imresize(shift_x,2,'nearest');
        shift_y = imresize(shift_y,2,'nearest');
    end

    It = zeros(ceil(m/step)*step+rongyu_h,ceil(n/step)*step+rongyu_h);
    It(rongyu_hh+1:m+rongyu_hh,rongyu_hh+1:n+rongyu_hh) = I1;
    I1 = It;
    
    It(rongyu_hh+1:m+rongyu_hh,rongyu_hh+1:n+rongyu_hh) = I2;
    I2 = It;
    clear It;
    shift_I1 = zeros(size(I1,1),size(I1,2));

    for i = 1:ceil(m/step)
        for j = 1:ceil(n/step)
            x_ = (i-1)*step+1;
            y_ = (j-1)*step+1;
            I11 = I1(x_:x_+step+rongyu_h-1,y_:y_+step+rongyu_h-1);
            I22 = I2(x_:x_+step+rongyu_h-1,y_:y_+step+rongyu_h-1);

%             if total_ == 132
%                 total_
%             end
            t_I11 = I11(rongyu_hh+1:step+rongyu_hh,rongyu_hh+1:step+rongyu_hh);
                t_I22 = I22(rongyu_hh+1:step+rongyu_hh,rongyu_hh+1:step+rongyu_hh);
            if max(t_I11(:)) <thresh_signal || max(t_I22(:)) < thresh_signal
                max_corr(i,j) = 1;
                %shift_x(i,j) = shift_x(i,j)+0;
                %shift_y(i,j) = shift_y(i,j)+0;
                total_ = total_+1;
                disp(strcat(datestr(now),32,32,num2str(total_),32,'no signal'));
                shift_I1(x_+rongyu_hh:x_+step+rongyu_h-1,y_++rongyu_hh:y_+step+rongyu_h-1) = ...
                    I11(rongyu_hh+1:rongyu_h+step,rongyu_hh+1:rongyu_h+step);
            else
                %t_I11 = I11(rongyu_hh+1:step+rongyu_hh,rongyu_hh+1:step+rongyu_hh);
                %t_I22 = I22(rongyu_hh+1:step+rongyu_hh,rongyu_hh+1:step+rongyu_hh);
                corr_in = normxcorr2(t_I11,t_I22);
                cor_t = max(corr_in(:)) ;

                if iter ~= 1 && ( cor_t> thresh_corr )% min(max_corr(i,j),thresh_corr) )
                    %shift_x(i,j) = shift_x(i,j)+0;
                    %shift_y(i,j) = shift_y(i,j)+0;
                    %figure,imshowpair(I11,I22);
                    total_ = total_+1;
                    max_corr(i,j) = cor_t;
                    disp(strcat(datestr(now),32,32,num2str(total_),32,'is higher'));
                    shift_I1(x_+rongyu_hh:x_+step+rongyu_h-1,y_++rongyu_hh:y_+step+rongyu_h-1) = ...
                        I11(rongyu_hh+1:rongyu_h+step,rongyu_hh+1:rongyu_h+step);
                    %figure,imshowpair(t_I11,t_I22);
                    %                     if cor_t> thresh_corr
                    %                         figure,imshowpair(I11,I22);
                    %                     end
                else
                    %t_I11 = max(I11-thresh_signal,0);
                    %t_I22 = max(I22-thresh_signal,0);
                    %corr_in = normxcorr2(t_I11,t_I22);
                    max_corr(i,j) = cor_t;
                    [shift_y_t,shift_x_t] = find(corr_in == max_corr(i,j));
                    if size(shift_x_t,1)>1 || size(shift_y_t,1)>1
                        shift_x_t = shift_x_t(1,1);
                        shift_y_t = shift_y_t(1,1);
                    end
                    shift_x_t = shift_x_t -  size(t_I11,1);
                    shift_y_t = shift_y_t -  size(t_I11,2);
                    if (shift_x_t^2+shift_y_t^2)>rongyu_hh^2
                        shift_x_t = 0;
                        shift_y_t = 0;
                        %max_corr(i,j) = 0.5;
                    end
                    a = shift_x(i,j)+shift_x_t/(2^(iter-1));
                    b = shift_y(i,j)+shift_y_t/(2^(iter-1));
                    if (a^2+b^2)<rongyu_hh^2
                        shift_x(i,j) = a;
                        shift_y(i,j) = b;
                        %max_corr(i,j) = 0.5;
                    end
                    I11 = imtranslate(I11,[shift_x_t,shift_y_t]);
                    %I11 = imtranslate(I11,[shift_x(i,j),shift_y(i,j)]);
                    total_ = total_+1;
                    %disp(strcat(datestr(now),32,32,num2str(total_),32,'is calculated'));
                    shift_I1(x_+rongyu_hh:x_+step+rongyu_h-1,y_++rongyu_hh:y_+step+rongyu_h-1) = ...
                        I11(rongyu_hh+1:rongyu_h+step,rongyu_hh+1:rongyu_h+step);
                end
            end
            
            %figure,imshowpair(I11,I22);
            
            %                 if max_corr(i,j) > 0.85
            %                     disp(strcat(datestr(now),32,32,num2str(total_),32,32,'corr_max is achieved at iteration =',32,num2str(iter),32,'with',32,num2str(max_corr(i,j))));
            %                     break
            %                 end
            %figure,imshowpair(I11,I22);
        end
    end
    tt(1:m,1:n) = shift_I1(rongyu_hh+1:m+rongyu_hh,rongyu_hh+1:n+rongyu_hh);
    shift_I1 = tt;
    %clear tt
    %imwrite(uint16(shift_I1),strcat('your/file/path/input/',num2str(iter),'pei_CH1_136.tif'));
    disp(strcat(datestr(now),32,32,num2str(iter),32,'is finished!'));
end
% 
disp(strcat(datestr(now),32,'nonlinear begins'));
thresh_ = 0.75;%thresh_corr;%0.75;
I1(rongyu_hh+1:rongyu_hh+m,rongyu_hh+1:rongyu_hh+n) = shift_I1;
[optimizer, metric] = imregconfig('multimodal');
final_ = zeros(size(I1,1),size(I1,2));
for i = 1:ceil(m/step)
    for j = 1:ceil(n/step)
        x_ = (i-1)*step+1;
            y_ = (j-1)*step+1;
        if max_corr(i,j)<thresh_
            I11 = I1(x_:x_+step+rongyu_h-1,y_:y_+step+rongyu_h-1);
            I22 = I2(x_:x_+step+rongyu_h-1,y_:y_+step+rongyu_h-1);
            tt = imregister(I11,I22,'affine',optimizer,metric);%'affine'
            final_(x_+rongyu_hh:x_+step+rongyu_h-1,y_++rongyu_hh:y_+step+rongyu_h-1) = ...
                tt(rongyu_hh+1:rongyu_h+step,rongyu_hh+1:rongyu_h+step);
            %figure,imshowpair(tt,I22);
        else
            I11 = I1(x_:x_+step+rongyu_h-1,y_:y_+step+rongyu_h-1);
            final_(x_+rongyu_hh:x_+step+rongyu_h-1,y_++rongyu_hh:y_+step+rongyu_h-1) = ...
                I11(rongyu_hh+1:rongyu_h+step,rongyu_hh+1:rongyu_h+step);
            %disp(strcat(datestr(now),32,'already'));
        end
        %figure,imshow(final_,[]);
    end
end
clear tt
tt(1:m,1:n) = final_(rongyu_hh+1:m+rongyu_hh,rongyu_hh+1:n+rongyu_hh);
%imwrite(uint16(tt),'your/file/path/l.tif');
disp(strcat(datestr(now),32,'program ends'));
end
% 
% %figure,imshow(I1(x_+rongyu_hh:x_+step+rongyu_h-1,y_+rongyu_hh:y_+step+rongyu_h-1),[]);
