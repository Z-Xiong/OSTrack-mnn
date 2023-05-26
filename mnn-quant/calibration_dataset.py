import numpy as np
import os
from PIL import Image
import MNN
F = MNN.expr


# adapted from pycaffe
def load_image(filename, color=True):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    filename : string
    color : boolean
        flag for color format. True (default) loads as RGB while False
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image : an image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or
        of size (H x W x 1) in grayscale.
    """
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def center_crop(image_data, crop_factor):
    height, width, channels = image_data.shape

    h_size = int(height * crop_factor)
    h_start = int((height - h_size) / 2)
    h_end = h_start + h_size

    w_size = int(width * crop_factor)
    w_start = int((width - w_size) / 2)
    w_end = w_start + w_size

    cropped_image = image_data[h_start:h_end, w_start:w_end, :]

    return cropped_image


def resize_image(image, shape):
    im = Image.fromarray(image)
    im = im.resize(shape)
    resized_image = np.array(im)

    return resized_image

def traverse_folder(folder_path):
    z_image_list = []
    x_image_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为图片文件
            if file.endswith(('.x.jpg')):
                x_image_path = os.path.join(root, file)
                # 在这里可以对图片路径进行处理或使用
                x_image_list.append(x_image_path)
            if file.endswith(('.z.jpg')):
                z_image_path = os.path.join(root, file)
                # 在这里可以对图片路径进行处理或使用
                z_image_list.append(z_image_path)
    assert len(z_image_list) == len(x_image_list)
    return z_image_list, x_image_list


class CalibrationDataset(MNN.data.Dataset):
    '''
    This is demo for Imagenet calibration dataset. like pytorch, you need to overload __getiterm__ and __len__ methods
    __getiterm__ should return a sample in F.const, and you should not use batch dimension here
    __len__ should return the number of total samples in the calibration dataset
    '''
    def __init__(self, image_folder):
        super(CalibrationDataset, self).__init__()
        self.image_folder = image_folder
        
        self.z_image_list, self.x_image_list = traverse_folder(image_folder)
        self.z_image_list = self.z_image_list[0:200]
        self.x_image_list = self.x_image_list[0:200]

    def __getitem__(self, index):
        z_image_name = self.z_image_list[index]
        # preprocess your data here, the following code are for tensorflow mobilenets
        image_data = load_image(z_image_name)
        image_data = resize_image(image_data, (128, 128))
        image_data = (image_data - (0.485*255, 0.456*255, 0.406*255)) * (1/(0.229*255), 1/(0.224*255), 1/(0.225*255))

        # after preprocessing the data, convert it to MNN data structure\
        image_data = image_data.transpose(2, 0, 1)
        z = F.const(image_data.flatten().tolist(), [3, 128, 128], F.data_format.NCHW, F.dtype.float)


        x_image_name = self.x_image_list[index]
        # preprocess your data here, the following code are for tensorflow mobilenets
        image_data = load_image(x_image_name)
        image_data = resize_image(image_data, (256, 256))
        image_data = (image_data - (0.485*255, 0.456*255, 0.406*255)) * (1/(0.229*255), 1/(0.224*255), 1/(0.225*255))

        # after preprocessing the data, convert it to MNN data structure
        image_data = image_data.transpose(2, 0, 1)
        x = F.const(image_data.flatten().tolist(), [3, 256, 256], F.data_format.NCHW, F.dtype.float)
        '''
        first list for inputs, and may have many inputs, so it's a list
        if your model have more than one inputs, add the preprocessed MNN const data to the input list

        second list for targets, also, there may be more than one targets
        for calibration dataset, we don't need labels, so leave it blank

        Note that, the input order in the first list should be the same in your 'config.yaml' file.
        '''
        
        return [z, x], []

    def __len__(self):
        # size of the dataset
        return len(self.z_image_list)


'''
initialize a CalibrationDataset object, its name should be exactly 'calibration_dataset'
'''
calibration_dataset = CalibrationDataset(image_folder='data/val')