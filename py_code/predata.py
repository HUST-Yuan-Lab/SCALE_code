import numpy as np
import os.path
import os
import tifffile as tiff
import cv2

# from percentnorm import percentile_normalization


class Processor:
    def __init__(self, file_root='', sample_number='', channels=(1,), strip_range=range(40, 41),
                 file_write='', point1=(0, 0), point2=(0, 0)):
        self.file_root = file_root
        self.sample_number = sample_number
        self.channels = channels
        self.strip_range = strip_range
        self.file_write = file_write
        self.p1 = point1
        # self.p2 = point2
        os.makedirs(self.file_write, exist_ok=True)
        self.file_list = []
        self.roi_enable = 1

    def get_image(self, layers, arrays):
        file_root = self.file_root
        sample_number = self.sample_number
        channels = self.channels
        strip_range = self.strip_range
        layer_range = layers
        file_write = self.file_write
        array_range = arrays
        roi = (self.p1[0], self.p1[1], self.p1[0] + self.p1[2], self.p1[1] + self.p1[3])
        print('image_roi', roi)
        os.makedirs(os.path.join(file_write, 'LiMo'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'nline'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'wide'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'line'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'LiMoc'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'LiMoe'), exist_ok=True)
        os.makedirs(os.path.join(file_write, 'LiMoce'), exist_ok=True)
        # os.makedirs(os.path.join(file_write, 'SR'), exist_ok=True)

        for ichannal in channels:
            for ilayer in layer_range:
                for istrip in strip_range:
                    image_list = []
                    for iarray in array_range:
                        image_name = sample_number + '_' + str('%05d' % ilayer) + '_' + str(
                            istrip) + '_CH' + str(ichannal) + '_SD1_' + str(iarray) + '.tif'
                        full_name = os.path.join(file_root, 'CH%d' % ichannal, sample_number + '_CH' + str(ichannal)
                                                 + '_' + str('%05d' % ilayer), image_name)
                        image = tiff.imread(full_name)
                        image = image[roi[0]:roi[2], roi[1]:roi[3]]     
                        image = image.astype('float32')                 
                        [h1, w1] = image.shape                          
                        image1 = image
                        # low, high, normalized_image = percentile_normalization(image1, 1, 99.9995)
                        # image1 = normalized_image*high
                        print('image_shape', image.shape)
                        image1 = np.clip(image1, 0, 65535)              
                        image_list.append(image1)                       

                    # SR = tiff.imread(os.path.join(file_root, 'SR', 'CH' + str(ichannal) + '_' + str(istrip) + '_SR' + '.tif'))
                    # print(SR.shape)
                    # SR = SR.astype('uint16')
                    # name_SR = (str('SR_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    # name_write_SR = os.path.join(file_write, str('SR'), name_SR)

                    nimage = np.stack(image_list, axis=0)               
                    print(nimage.shape)
                    nimage = nimage.astype('uint16')
                    name_nline = (str('nline_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_nline = os.path.join(file_write, str('nline'), name_nline)
                    tiff.imwrite(name_write_nline, nimage)              


                    line = nimage[2]                                    
                    name_line = (str('line_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_line = os.path.join(file_write, str('line'), name_line)
                    tiff.imwrite(name_write_line, line)

                    wide = np.mean(nimage, axis=0)                      
                    wide = wide.astype('uint16')
                    name_wide = (str('wide_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_wide = os.path.join(file_write, str('wide'), name_wide)
                    tiff.imwrite(name_write_wide, wide)

                    limo = self.get_limo(image_list)                    
                    name_subline = (str('LiMo_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_limo = os.path.join(file_write, 'LiMo', name_subline)
                    tiff.imwrite(name_write_limo, limo)

                    limoc = self.get_limoc(image_list)  
                    name_sublinec = (str('LiMoc_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_limoc = os.path.join(file_write, 'LiMoc', name_sublinec)
                    tiff.imwrite(name_write_limoc, limoc)

                    limoe = self.get_limoe(image_list)  
                    name_sublinee = (str('LiMoe_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_limoe = os.path.join(file_write, 'LiMoe', name_sublinee)
                    tiff.imwrite(name_write_limoe, limoe)
                    #
                    limoce = np.stack((limoc, limoe), axis=0)  
                    name_sublinece = (str('LiMoce_%05d_%02d_CH%d.tif' % (ilayer, istrip, ichannal)))
                    name_write_limoce = os.path.join(file_write, 'LiMoce', name_sublinece)
                    tiff.imwrite(name_write_limoce, limoce)


    def image2data(self, file_name='test', roi=(), shape=(240, 240),  red=4, sigma=0):
        roi = (roi[0], roi[1], roi[2], roi[3])
        print('roi', roi)
        sh = (shape[0], shape[1])
        print('sh', sh)
        step = (sh[0] - sh[0] // red, sh[1] - sh[1] // red)     
        print('step', step)
        file = os.path.join(self.file_write, file_name)
        print(file)
        name_list = self.get_name(file)                         
        print(len(name_list))
        list_range = range(len(name_list))                      
        data_list = []
        for i in list_range:
            name = os.path.join(file, name_list[i])
            image = tiff.imread(name)
            print(name)
            print(image.shape)
            if image.ndim == 2:
                image = image[np.newaxis, :, :]
            image = image[:, roi[0]:roi[2]+roi[0], roi[1]:roi[3]+roi[1]]
            # temp = self.image_split(image, sh, step)
            temp = self.image_split(image, sh, step)
            data_list.append(temp)  
            # temp_rotated = [np.rot90(slice_, 1, axes=(1, 2)) for slice_ in temp]
        print(len(data_list))
        data = np.vstack(data_list)
        os.makedirs(os.path.join(self.file_write, 'Data'), exist_ok=True)
        name = os.path.join(self.file_write, 'Data', file_name + '.npy')
        np.save(name, data)
        return data

    def get_name(self, file_dir):
        files = []
        for root, dirs, files in os.walk(file_dir):
            print(str('该文件下有%d个文件' % len(files)))  
        return files

    def get_limo(self, image_list):
        i = image_list
        limo = (i[2]+i[3])*2-i[0]-i[1]-i[4]-i[5]
        # limo = (i[4] + i[5] + i[6] + i[7]) * 2 - i[0] - i[1] - i[2] - i[3] - i[8] - i[9] - i[10] - i[11]
        limo = (abs(limo) + limo)/2
        limo = ((65535 + limo)-abs(65535 - limo))/2
        limo = limo.astype('uint16')
        return limo

    def get_limoc(self, image_list):
        i = image_list
        limoc = (i[2]+i[3])/2  #-i[0]-i[1]-i[4]-i[5]
        # limoc = (i[4] + i[5] + i[6] + i[7]) / 4
        limoc = (abs(limoc) + limoc)/2
        limoc = ((65535 + limoc)-abs(65535 - limoc))/2
        limoc = limoc.astype('uint16')
        return limoc

    def get_limoe(self, image_list):
        i = image_list
        limoe = (i[0]+i[1]+i[4]+i[5])/4
        # limoe = (i[0] + i[1] + i[2] + i[3] + i[8] + i[9] + i[10] + i[11]) / 8
        limoe = (abs(limoe) + limoe)/2
        limoe = ((65535 + limoe)-abs(65535 - limoe))/2
        limoe = limoe.astype('uint16')
        return limoe

    def image_split(self, image, shape=(1, 1), step=(1, 1)):
        num_x = shape[0]
        num_y = shape[1]
        step_x = step[0]
        step_y = step[1]
        if np.ndim(image) == 2:
            [w, h] = np.shape(image)
            image = np.reshape(image, (1, w, h))        
            print(image.shape)
        [d, w, h] = np.shape(image)
        imagelist = []
        for i in range(0, w - num_x + 1, step_x):
            for j in range(0, h - num_y + 1, step_y):   
                image_temp = image[:, i:i + num_x, j:j + num_y]
                imagelist.append(image_temp)
        data = np.stack(imagelist, axis=0)
        print(data.shape)
        return data