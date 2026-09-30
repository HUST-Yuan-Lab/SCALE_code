import numpy as np
import torch
# from model import UNet
# from RCAN import RCAN
from URCANsub import URCANsub
# from newrcan import RCAN
import tifffile as tiff
import time
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from image_split_and_merge import *

start = time.time()     
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
device = torch.device('cuda:0')
# model = UNet(n_channel_in=1, n_channel_out=1, n_filter_base=64).to(device)
# model = RCAN(num_channels=64,
#                  num_residual_blocks=5,
#                  num_residual_groups=10,
#                  channel_reduction=16).to(device)
model = URCANsub(1, 1, 32, 5).to(device)
# model = RCAN(10, 64).to(device)

channels = [1]
scale = 12000    
layers = range(1, 2, 1)
strips = range(40, 48, 1)
# array_range = range(1, 7)
patch_size = 288
stride = 144

model.load_state_dict(
    torch.load(r'your/file/path/model.pth'))
model.eval()

file_root = r'your/file/path/input'




def get_wfimage(ilayer, istrip):
    image_list = []

    # for iarray in array_range:
    #     image_name = '250604_' + str('%05d' % ilayer) + '_' + str(istrip) + '_CH%d_SD1_' % ch + str(iarray) + '.tif'
    #     full_name = os.path.join(file_root,  'CH%d' % ch, '250604' + '_CH' + str(ch) + '_' + str('%05d' % ilayer), image_name)
    #     print(full_name)
    #     image = tiff.imread(full_name)
    #     image = image.astype('float32')
    #     image1 = image
    #     image_list.append(image1)
    #
    # nimage = np.stack(image_list, axis=0)
    # nimage = np.clip(nimage, 0, 65535)
    # nimage = nimage.astype('uint16')
    # # wide = nimage[5]
    # wide = wide.astype('uint16')
    # print(wide.shape)
    # # wffile_write = os.path.join(file_root, 'your/file/path/output', 'wf', 'CH%d' % ch, 'sample_CH%d_' % ch + str('%05d' % ilayer))
    # # # wffile_write = os.path.join(r'your/file/path/output')
    # # os.makedirs(wffile_write, exist_ok=True)
    # # wffile_write1 = os.path.join(wffile_write, 'wf_250224_%05d_%02d_CH%d.tif' % (ilayer, istrip, ch))
    # # tiff.imwrite(wffile_write1, np.rot90(wide, 3))

    image_name = 'wide_' + str('%05d' % ilayer) + '_' + str(istrip) + '_CH' + str(channels[0]) + '.tif'
    full_name = os.path.join(file_root,  image_name)
    print(full_name)
    image = tiff.imread(full_name)
    wide = image.astype('float32')
    wide = np.clip(wide, 0, 65535)
    wide = wide.astype('uint16')
    wide = np.rot90(wide, 1)        
    print(wide.shape)
    return wide

def image2data(image):
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    # image_stack.shape (32, 7950, 1960)
    data, _, r, c, before1, before2, after1, after2 = image2patch(image, patch_size=patch_size, stride=stride)
    return data, r, c, before1, before2, after1, after2


class MyDataset(Dataset):
    def __init__(self, inputs, scale):
        self.input = inputs
        self.scale = scale

    def __len__(self):
        length, _, _, _ = self.input.shape
        return length

    def __getitem__(self, idx):
        return self.input[idx, :, :, :].astype('float32') / self.scale


def get_data(ilayer, istrip):
    image_stack = get_wfimage(ilayer, istrip)
    # image_stack = np.vstack([image_stack1, image_stack2])
    # image_stack1 = 0
    # image_stack2 = 0
    # image_stack = image_stack1
    print('image_stack.shape', image_stack.shape)
    data, r, c, before1, before2, after1, after2 = image2data(image_stack)
    print('data.shape', data.shape)
    image_stack = 0
    data_loader = DataLoader(MyDataset(data, scale), batch_size=80)
    return data_loader, r, c, before1, before2, after1, after2


def image_pre(model, data, device):
    model.eval()
    with torch.no_grad():
        data_list = []
        for x in data:
            x_in = x.to(device)
            x_out = model(x_in)
            x_numpy = x_out.cpu().detach().numpy()
            data_list.append(x_numpy)
        data_pre = np.vstack(data_list)
        # x_in = data.to(device)
        # x_out = model(x_in)
        # x_numpy = x_out.cpu().detach().numpy()
        # data_pre = x_numpy
    return data_pre

def data2image(data, r, c, before1, before2, after1, after2):
    # data.shape (32, 5, 256, 1952)
    ch = data.shape[1]
    data[data >= 1] = 1
    data[data <= 0] = 0
    data = data * scale
    image = patch2image(data, ch, r, c, before1, before2, after1, after2, patch_size=patch_size, stride=stride)
    print('image.shape', image.shape)
    # image.shape (5, 7950, 1952)
    return image

def output_stack():
    # for ch in channels:
    #     for ilayer in layers:
    #         for istrip in strips:
    #             data_loader, r, c, before1, before2, after1, after2 = get_data(ilayer, istrip, ch)
    #             tstar = time.time()
    #             data_pre = image_pre(model, data_loader, device)
    #             tend = time.time()
    #             print('data_pre.shape', data_pre.shape)
    #             image = data2image(data_pre, r, c, before1, before2, after1, after2)
    #             print('image_pre.shape', image.shape)
    #             # file_write = os.path.join(file_root, 'your/file/path/output', 'VCLS_Limoce', 'CH%d' % ch, 'sample_CH%d_' % ch + str('%05d' % ilayer))
    #             # os.makedirs(file_write, exist_ok=True)
    #             # file_write1 = os.path.join(file_write, 'VCLS_limoce_%05d_%02d_CH%d.tif' % (ilayer, istrip, ch))
    #             # file_write = os.path.join(file_root, 'result_3D', f'CH{ch}', f'sample_CH{ch}_{ilayer:05d}')
    #             file_write = r'your/file/path/output'
    #             os.makedirs(file_write, exist_ok=True)
    #             file_name = f'yLC_{ilayer:05d}_{istrip:02d}.tif'
    #             file_write1 = os.path.join(file_write, file_name)
    #             tiff.imwrite(file_write1, image)
    #             data_loader = 0
    #             data_pre = 0
    #             image = 0


    for ilayer in layers:
        for istrip in strips:
            data_loader, r, c, before1, before2, after1, after2 = get_data(ilayer, istrip)
            tstar = time.time()
            data_pre = image_pre(model, data_loader, device)
            tend = time.time()
            print('总:%s ' % (tend - tstar))
            print('data_pre.shape', data_pre.shape)
            image = data2image(data_pre, r, c, before1, before2, after1, after2)
            image = np.rot90(image, 3, (1, 2))  
            print('image_pre.shape', image.shape)
            # file_write = os.path.join(file_root, 'your/file/path/output', 'VCLS_Limoce', 'CH%d' % ch, 'sample_CH%d_' % ch + str('%05d' % ilayer))
            # os.makedirs(file_write, exist_ok=True)
            # file_write1 = os.path.join(file_write, 'VCLS_limoce_%05d_%02d_CH%d.tif' % (ilayer, istrip, ch))
            # file_write = os.path.join(file_root, 'result_3D', f'CH{ch}', f'sample_CH{ch}_{ilayer:05d}')
            file_write = r'your/file/path/output'
            os.makedirs(file_write, exist_ok=True)
            file_name = f'y_lc_{ilayer:05d}_{istrip:02d}_CH{channels[0]}.tif'
            file_write1 = os.path.join(file_write, file_name)
            tiff.imwrite(file_write1, image)
            data_loader = 0
            data_pre = 0
            image = 0


output_stack()
end = time.time()
print('Time:{:4f}s'.format(end-start))