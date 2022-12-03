# -*-  coding = utf-8 -*-
# @Time : 2022/10/9 9:51 下午
# @author : Wang Zhixian
# @File : data_load.py
# @Software: PyCharm

import numpy as np
import tqdm
import os

import SimpleITK as sitk
from tqdm import tqdm

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))[::-1, ...][:, ::-1]



def trainImageFetch(images_id, train_image_dir, train_mask_dir, img_size):
    image_train = np.zeros((len(images_id), *img_size), dtype=np.float32)
    mask_train = np.zeros((len(images_id), *img_size), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(images_id), total=len(images_id)):
        image_path = os.path.join(train_image_dir, "image"+image_id+'.nii.gz')
        mask_path = os.path.join(train_mask_dir, "image"+image_id+'.nii.gz')

        image = read_img(image_path).astype(np.float32)
        mask = read_img(mask_path).astype(np.float32)

        image_train[idx] = image
        mask_train[idx] = mask

    return image_train, mask_train

def testImageFetch(test_id,test_image_dir,test_mask_dir,img_size):
    image_test = np.zeros((len(test_id), *img_size), dtype=np.float32)
    mask_test = np.zeros((len(test_id), *img_size), dtype=np.float32)

    for idx, image_id in tqdm(enumerate(test_id), total=len(test_id)):
        image_path = os.path.join(test_image_dir, "image"+image_id+'.nii.gz')
        mask_path = os.path.join(test_mask_dir, "image"+image_id+'.nii.gz')

        image = read_img(image_path).astype(np.float32)
        mask = read_img(mask_path).astype(np.float32)

        image_test[idx] = image
        mask_test[idx] = mask

    return image_test, mask_test