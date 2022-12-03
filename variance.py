# -*-  coding = utf-8 -*-
# @Time : 2022/10/11 9:43 上午
# @author : Wang Zhixian
# @File : variance.py
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
from tqdm import tqdm

import SimpleITK as sitk
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mp
import os

from math import gamma as gamma

train_image_dir = test_image_dir = 'data'
train_mask_dir = test_mask_dir = 'mask'

def variance(net, device):
    test_id = []
    for i in range(5):
        i = i + 1 + 34
        test_id.append('00{i}'.format(i=i))
    X_test, y_test = trainImageFetch(test_id, test_image_dir, train_mask_dir, img_size)
    test_data = qubiqDataset(X_test, 'val', y_test, pad_left=0, pad_right=0)
    train_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    n_test = len(train_loader)
    k = 0
    t = 256
    for batch, image in enumerate(train_loader):
        data = image[0].to(device=device, dtype=torch.float32)
        label = image[1].to(device=device, dtype=torch.float32)[0,0,:,:]
        alpha = torch.exp(net(data))[0, :, :, :]
        #alpha = F.softmax(unet(data), dim=1)[0, :, :, :]
        var = torch.zeros(t, t)

        for i in range(256):
            for j in range(256):
                #a = Dirichlet(alpha[:, i, j])
                #samp = torch.ones(200)
                #for q in range(200):
                    #samp[q] = a.sample()[0]

        #alpha0 = alpha[0, :, :]+alpha[1, :, :]
        #var = (alpha[0, :, :]*alpha[0, :, :])/((alpha0**2)*(alpha0+1))
                #var[i, j] = torch.var(a.sample(torch.Size([100]))[:, 0])
                if label[i, j]==1:
                    a0 = alpha[0, i, j]
                    a1 = alpha[1, i, j]
                else:
                    a1 = alpha[0, i, j]
                    a0 = alpha[1, i, j]

                #print(gamma(a0+2)*gamma(a1)*gamma(a0)*gamma(a1+1)*(a0+a1+1))
                #var[i, j] = (gamma(a0+2)*gamma(a1)*gamma(a0)*gamma(a1+1)*(a0+a1+1)-(gamma(a0+1)*gamma(a1+1))*(gamma(a0+1)*gamma(a1+1)))
                #var[i, j] = var[i, j]/((a0+a1+1)*(a0+a1+1)*(gamma(a0)*gamma(a1+1))*(gamma(a0)*gamma(a1+1)))
                E = a0/(a0+a1)
                E2 = a0*(a0+1)/((a0+a1+1)*(a0+a1))
                E3 = a0*(a0+1)*(a0+2)/((a0+a1+1)*(a0+a1)*(a0+a1+2))
                var[i, j] = (E2-E3)/(1-E) - ((E-E2)/(1-E))**2




        #var = var
        var = var.detach().numpy()#[60:256, 60:256]
        #fig = plt.figure()
        #sns_plot = sns.heatmap(var)
        #plt.savefig('000'+test_id[k]+'.png')
        #plt.show()
        output_out = sitk.GetImageFromArray(var)
        if os.path.isdir('./var')!=True:
            os.mkdir('./var')
        sitk.WriteImage(output_out, 'var/image' + test_id[k] + '_var.nii.gz')
        #mp.imsave('000'+test_id[k]+'.png', sns_plot)
        print(k)
        k = k+1




if __name__ == '__main__':
    img_size = (256, 256)
    unet = torch.load('./result/net.pkl', map_location='cpu')
    device = torch.device('cpu')
    variance(unet, device)


