from predata import *


def predata_main():
    file_root = r'your/file/path/input'      
    sample_number = '250108'            
    channels = (1,)             
    arrays = range(5, 11)                
    strip_range = range(40, 48, 1)      
    point_1 = (0, 0, 10000, 2048)    
    layers = range(1, 2, 1)             
    save_file = r'your/file/path/dataset'  
    p1 = (point_1[0], point_1[1], point_1[2], point_1[3])  
    print(p1)

    get_image = 10
    get_data = [0, 0, 0, 0, 0, 0, 0]    

    point1 = p1[0:4]

    pp1 = Processor(file_root, sample_number, channels, strip_range, save_file, point1, point1)  

    if get_image == 10:
        pp1.get_image(layers, arrays)

    roi = (0, 0, 10000, 2048)

    shape = (256, 256)
    red = 4  

    if get_data[0] == 1:
        # sigma = 0
        file_name = 'LiMo'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print(data.shape)
        print('LiMo_data', data.shape)

    if get_data[1] == 1:
        # sigma = 1
        file_name = 'nline'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print('nline_data', data.shape)

    if get_data[2] == 1:
        # sigma = 2
        file_name = 'wide'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print('wide_data', data.shape)

    if get_data[3] == 1:
        # sigma = 3
        file_name = 'line'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print('line_data', data.shape)

    if get_data[4] == 1:
        # sigma = 0
        file_name = 'LiMoc'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print(data.shape)
        print('LiMoc_data', data.shape)

    if get_data[5] == 1:
        # sigma = 0
        file_name = 'LiMoe'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print(data.shape)
        print('LiMoe_data', data.shape)

    if get_data[6] == 1:
        # sigma = 0
        file_name = 'LiMoce'
        data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
        print(data.shape)
        print('LiMoce_data', data.shape)

    # if get_data[7] == 1:
    #     # sigma = 0
    #     file_name = 'SR'
    #     data = pp1.image2data(file_name=file_name, roi=roi, shape=shape, red=red)
    #     print(data.shape)
    #     print('SR_data', data.shape)


predata_main()
