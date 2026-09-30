function [recon] = dual_view_deconv_sparse(ali,ref,filt1,filt2,iter)


recon = zeros(size(ali,1),size(ali,2),size(ali,3));
for i = 1:size(ali,3)
    tmp1 = double(ali(:,:,i));
    tmp2 = double(ref(:,:,i));
    if max(tmp1(:)) == 0 && max(tmp2(:)) == 0
        recon(:,:,i) = tmp1;
        %disp(strcat(datestr(now),': layer',32,32,num2str(i),32,32,'no data'));
    else
        max_ = max(tmp1(:));
        for j = 1:iter
            if j == 1
                %tmp1 = deconvL2(tmp1,filt1,0.001,1);
                %tmp2 = deconvL2(tmp2,filt2,0.001,1);
                tmp1 = tmp1+tmp2;
                tmp1 = tmp1/2;
                tmp2 = tmp1;
                recon(:,:,i) = tmp1;
            else
                
                if sum(sum(isnan(tmp1)))>0 || sum(sum(isnan(tmp2)))>0
                    recon(:,:,i) = tmp1;
                    %disp(strcat(datestr(now),': layer',32,32,num2str(i),32,32,'deconvL2 no data'));
                    break
                else
                    RL1 = deconvlucy(tmp1,filt1,1);
                    RL2 = deconvlucy(tmp2,filt2,1);
                    if sum(sum(isnan(RL1)))>0 || sum(sum(isnan(tmp2)))>0
                        recon(:,:,i) = tmp1;
                        %disp(strcat(datestr(now),': layer',32,32,num2str(i),32,32,'deconvlucy no data'));
                        break
                    else
                        RL1 = RL1./max(RL1(:))*max_;
                        RL2 = RL2./max(RL2(:))*max_;
                        %RL = max(RL1,RL2);
                        RL = (RL1+RL2)/2;
                        tmp1 = RL;
                        tmp2 = RL;
                        recon(:,:,i) = RL;
                    end
                end
            end
        end
    end
end
end