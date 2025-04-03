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
    #dataloader,device=get_dataloader(config)
    device=torch.device("cpu")
    netD=Discriminator(config).to(device)
    netG=Generator(config).to(device)
    netG.apply(init_weights)
    netD.apply(init_weights)
    print(netD)
    print(netG)


