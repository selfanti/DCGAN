#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import yaml
from model import Generator,Discriminator
from dataloader import get_dataloader
# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results
def load_config(path):
    with open(path,'r') as file:
        config=yaml.safe_load(file)
    return config

def init_weights(model):
    classname=model.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(model.weight.data,0.0,0.02)
    elif classname.find('BatchNorm')!=-1:
        nn.init.normal_(model.weight.data,1.0,0.02)
        nn.init.constant_(model.bias.data,0)


if __name__ =="__main__":
    config=load_config(r"U:\Users\Enlink\PycharmProjects\DCGAN\config\config.yaml")
    dataloader,device=get_dataloader(config)
    netD=Discriminator(config).to(device)
    netG=Generator(config).to(device)
    netG.apply(init_weights)
    netD.apply(init_weights)
    print(netD)
    print(netG)
    criterion=nn.BCELoss()
    nz=config['nz']
    lr=config['lr']
    fixed_noise=torch.randn(64,100,1,1,device=device)
    real_label=1
    fake_label=0
    beta1=config['beta1']
    num_epochs=config['num_epochs']
    optimizer_D=optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
    optimizer_G=optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))

    img_list=[]
    G_losses=[]
    D_losses=[]
    iters=0
    print("Starting Training Loop")

    #训练的目的是让判别器（Discriminator）对虚假数据的输出尽可能接近0，对真实数据的输出尽可能接近1.
    #对于生成器，需要尽可能的提高欺骗判别器的能力，最终是让D(G(z))收敛到0.5
    for epoch in range(num_epochs):
        for i,data in enumerate(dataloader,0):
            netD.zero_grad()
