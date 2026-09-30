import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tifffile as tiff

# from percentnorm import percentile_normalization


def load_data(file_name):
    tr_i = os.path.join(file_name, 'wide.npy')      
    tr_o = os.path.join(file_name, 'line.npy')    

    input0 = np.load(tr_i)
    output0 = np.load(tr_o)
    print('inputs:', input0.shape)
    print('outputs:', output0.shape)

    indexs = []
    for m in range(output0.shape[0]):
        output_index = output0[m]
        input_index = input0[m]
        if output_index.max() > 60 and output_index.mean() > 0.01 and input_index.max() > 0:
            indexs.append(m)

    input_choose = input0[indexs]
    output_choose = output0[indexs]

    print('input_choose shape:', input_choose.shape)
    print('output_choose shape:', output_choose.shape)

    return input_choose, output_choose

class MyDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.input = inputs
        self.output = outputs

    def __len__(self):
        return self.output.shape[0]

    def __getitem__(self, idx):
        # scale_out = percentile_normalization(self.output[idx].astype('float32'), 0.01, 99.995)
        return self.input[idx].astype('float32') / 12000, self.output[idx].astype('float32') / 12000


def my_dataloader(file, ratio, batchsize=16, shuffle=True):          
    inputs, outputs = load_data(file)

    print('inputs:', inputs.shape)
    print('outputs:', outputs.shape)

    length = outputs.shape[0]

    num_train = int(length * ratio[0])
    num_test = int(length * ratio[1])
    num_val = int(length * ratio[2])

    train_inputs = inputs[:num_train]
    train_outputs = outputs[:num_train]

    test_inputs = inputs[num_train:num_train + num_test]
    test_outputs = outputs[num_train:num_train + num_test]

    val_inputs = inputs[num_train + num_test:num_train + num_test + num_val]
    val_outputs = outputs[num_train + num_test:num_train + num_test + num_val]

    for i in range(16):
        os.makedirs(r'your/file/path/dataloadercheck', exist_ok=True)
        tiff.imwrite(r'your/file/path/dataloadercheck/input_wide_%d.tif' % i, train_inputs[40 * i].astype('uint16'))
        tiff.imwrite(r'your/file/path/dataloadercheck/output_line_%d.tif' % i, train_outputs[40 * i].astype('uint16'))
    print('done')

    train_dataloader = DataLoader(MyDataset(train_inputs, train_outputs), batch_size=batchsize, drop_last=True, shuffle=shuffle)
    test_dataloader = DataLoader(MyDataset(test_inputs, test_outputs), batch_size=batchsize, drop_last=False, shuffle=False)
    val_dataloader = DataLoader(MyDataset(val_inputs, val_outputs), batch_size=batchsize, drop_last=False, shuffle=False)

    loader = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    print('Train shape:', train_inputs.shape, train_outputs.shape)
    print('Test shape:', test_inputs.shape, test_outputs.shape)
    print('Val shape:', val_inputs.shape, val_outputs.shape)

    return loader


# if __name__ == '__main__':
#     data_dir = r'your/file/path/dataset'
#     loader = my_dataloader(data_dir, [0.8, 0.1, 0.1], batchsize=4)
#     a = loader['train'].dataset.input.shape