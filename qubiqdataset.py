# -*-  coding = utf-8 -*-
# @Time : 2022/10/9 9:46 下午
# @author : Wang Zhixian
# @File : data_loader.py
# @Software: PyCharm


from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

from copy import deepcopy

class qubiqDataset(Dataset):
    def __init__(self, image_list, mode, mask_list=None, fine_size=256, pad_left=0, pad_right=0):
        self.imagelist = image_list
        self.mode = mode
        self.masklist = mask_list
        self.fine_size = fine_size
        '''self.pad_left = pad_left
        self.pad_right = pad_right'''

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        image = deepcopy(self.imagelist[idx])

        if self.mode == 'train':
            mask = deepcopy(self.masklist[idx])
            #label = np.where(mask.sum() == 0, 1.0, 0.0).astype(np.float32)

            '''if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image, mask = do_center_pad2(image, mask, self.pad_left, self.pad_right)'''

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask#, label

        elif self.mode == 'val':
            mask = deepcopy(self.masklist[idx])

            '''if self.fine_size != image.shape[0]:
                image, mask = do_resize2(image, mask, self.fine_size, self.fine_size)

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)'''

            image = image.reshape(1, image.shape[0], image.shape[1])
            mask = mask.reshape(1, mask.shape[0], mask.shape[1])

            return image, mask

        elif self.mode == 'test':
            '''if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))

            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)'''

            image = image.reshape(1, image.shape[0], image.shape[1])

            return image

