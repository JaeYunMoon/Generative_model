import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import numpy as np 

EPOCH = 200
BATCH_SIZE = 100 
LEARNING_RATE= 0.0002
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1 
LATENT_DIM = 100 
GT_SIZE = 10 # MNIST 

DIRECTORY_NAME = "./CGAN_RESULT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device {DEVICE}")
# Result Directory 
if not os.path.exists(DIRECTORY_NAME):
    os.makedirs(DIRECTORY_NAME)

# Model - discriminator 는 gan보다 layer가 하나더 많음 
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(IMAGE_SIZE*IMAGE_SIZE*IMAGE_CHANNEL+GT_SIZE,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        output = self.model(x)
        return output 

