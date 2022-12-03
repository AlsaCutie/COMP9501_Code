# -*-  coding = utf-8 -*-
# @Time : 2022/10/10 3:00 下午
# @author : Wang Zhixian
# @File : test.py
# @Software: PyCharm

import torch
import torch.optim as optim
from data_load import trainImageFetch
from qubiqdataset import qubiqDataset
from network import UNet
from torch.utils.data import DataLoader
from setalpha import target_alpha
from torch.distributions.dirichlet import Dirichlet
import torch.distributions as dist
from dice_loss import diceCoeff, getBinaryTensor
import torch.nn.functional as F
import wandb
import logging
import SimpleITK as sitk

train_image_dir = test_image_dir = 'data'
train_mask_dir = test_mask_dir = 'mask'


def eval(net, device):
    dice_loss = 0.0
    net.eval()
    test_id = []
    for i in range(5):
        i = i + 1 + 34
        test_id.append('00{i}'.format(i=i))
    X_test, y_test = trainImageFetch(test_id, test_image_dir, train_mask_dir, img_size)
    test_data = qubiqDataset(X_test, 'val', y_test, pad_left=0, pad_right=0)
    train_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    n_test = len(train_loader)
    k = 0
    for batch, image in enumerate(train_loader):
        data = image[0].to(device=device, dtype=torch.float32)
        label = image[1].to(device=device, dtype=torch.int)[0, 0, :, :]
        output = getBinaryTensor(torch.Tensor(unet(data)))[0, :, :]
        dice_loss = dice_loss + diceCoeff(label, output)
        output_out = sitk.GetImageFromArray(output)
        sitk.WriteImage(output_out, 'result/image'+test_id[k]+'_result.nii.gz')
        k = k+1
        print(k)
    print('dice_Loss: {:.6f}'.format(dice_loss / n_test))
    # print(n_test)
    # net.train()





if __name__ == '__main__':
    img_size = (256, 256)
    unet = torch.load('./result/net.pkl', map_location='cpu')
    device = torch.device('cpu')
    eval(unet, device)
