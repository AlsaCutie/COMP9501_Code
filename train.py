# -*-  coding = utf-8 -*-
# @Time : 2022/10/16 9:02 下午
# @author : Wang Zhixian
# @File : train_reshape.py
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
import os

train_image_dir = test_image_dir = 'data'
train_mask_dir = test_mask_dir = 'mask'


def train_net(net,
              device,
              epochs: int = 30,
              batch_size: int = 4,
              learning_rate: float = 0.002,
              save_checkpoint: bool = True):
    train_id = []
    for i in range(34):
        i = i + 1
        if (i <= 9):
            train_id.append('000{i}'.format(i=i))
        else:
            train_id.append('00{i}'.format(i=i))
    test_id = []
    for i in range(5):
        i = i + 1 + 34
        test_id.append('00{i}'.format(i=i))
    X_train, y_train = trainImageFetch(train_id, train_image_dir, train_mask_dir, img_size)
    train_data = qubiqDataset(X_train, 'train', y_train, pad_left=0, pad_right=0)

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    n_train = len(train_loader)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint))

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
        ''')
    t = 256
    global_step = 0
    print(img_size)
    for epoch in range(epochs):
        loss_total = 0.0
        net.train()
        dice_total = 0.0
        #with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch, image in tqdm(enumerate(train_loader), total=n_train):
            data = image[0].to(device=device, dtype=torch.float32)
            label = image[1][: , 0, :, :].to(device=device, dtype=torch.int)
            optimizer.zero_grad()
            output = torch.exp(net(data))
            target = label.permute(1, 2, 0).reshape(-1, 1)
            output_alpha = output.permute(2, 3, 0, 1).reshape(-1, 2)
            target = target.cpu()
            target_a = target_alpha(target)
            target_a = target_a.to(device=device, dtype=torch.float32)
            dirichlet1 = Dirichlet(output_alpha)
            dirichlet2 = Dirichlet(target_a)
            loss = torch.mean(dist.kl.kl_divergence(dirichlet1, dirichlet2))
            loss_total = loss_total + loss.item()
            loss.backward()
            dice_loss = diceCoeff(getBinaryTensor(torch.exp(net(image[0].to(device=device, dtype=torch.float32)))),
                                      image[1].to(device=device, dtype=torch.int))
            dice_total = dice_total + dice_loss
            optimizer.step()
            #pbar.update(data.shape[0])
            global_step += 1
            experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
            print(global_step)
                #pbar.set_postfix(**{'loss (batch)': loss_b.item()})
        print('Train Epoch: {} \t Loss: {:.6f}'.format(epoch, loss_total))
        print('Train Epoch: {} \t dice_Loss: {:.6f}'.format(epoch, dice_total / n_train))
    if os.path.isdir('./result')!=True:
        os.mkdir('./result')
    torch.save(net, './result/net.pkl')
    logging.info(f'Checkpoint saved!')


if __name__ == '__main__':
    epochs = 40
    device = torch.device('cuda')
    learning_rate = 0.002
    img_size = (256, 256)

    unet = UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
    train_net(unet, device, epochs=epochs)