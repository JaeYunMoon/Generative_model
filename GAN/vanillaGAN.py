import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time 
from torchvision.utils import save_image

EPOCH = 200
BATCH_SIZE = 100 
LEARNING_RATE= 0.0002
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1 
LATENT_DIM = 100 

DIRECTORY_NAME = "./GAN_RESULT"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Result Directory 
if not os.path.exists(DIRECTORY_NAME):
    os.makedirs(DIRECTORY_NAME)

# Model - Discriminator, Generator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(IMAGE_CHANNEL* IMAGE_SIZE*IMAGE_SIZE,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self,img):
        flattend = img.view(img.size,-1)
        output = self.model(flattend)
        return output 

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM,128),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,IMAGE_CHANNEL*IMAGE_SIZE*IMAGE_SIZE),
            nn.Tanh()
        )

    def forward(self,z):
        img = self.model(z)
        img = img.view(img.size(0),1,28,28)
        return img 

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
# Dataset 
transforms_train = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = torchvision.datasets.MNIST(root="./dataset", 
                                           train=True, 
                                           download=True, 
                                           transform=transforms_train)
dataloader = torch.utils.data.DataLoader(train_dataset, 
                                         batch_size=128, 
                                         shuffle=True, 
                                         num_workers=0)

# Loss Function 
adversarial_loss = nn.BCELoss().to(DEVICE)

# optimizer Function 
optimizer_G = torch.optim.Adam(generator.parameters(),
                               lr = LEARNING_RATE, 
                               betas = (0.5,0.999))

optimizer_D = torch.optim.Adam(discriminator.parameters(),
                               lr=LEARNING_RATE,
                               betas=(0.5,0.999))
# Training 
sample_interval = 2000 # 몇 번의 배치(batch)마다 결과를 출력할 것인지 설정 
start_time = time.time
for epoch in range(EPOCH):
    for i,(imgs,_) in enumerate(dataloader):
        # real Image, Fake Image에 대한 GT 생성 
        real = torch.cuda.FloatTensor(imgs.size(0),1).fill_(1.0)
        fake = torch.cuda.FloatTensor(imgs.size(0),1).fill_(0.0)

        real_imgs = imgs.to(DEVICE)

        ## generator Training 
        optimizer_G.zero_grad()
        # random noise sample - Fake image 
        z = torch.normal(mean = 0, std=1,
                         size=(imgs.shape[0],LATENT_DIM)).to(DEVICE)
        # image generate
        generatedImg = generator(z)
        # generator loss calculate 
        gLoss = adversarial_loss(discriminator(generatedImg),real)

        # generator Backword 
        gLoss.backward()
        optimizer_G.step() 
    
        ## Discriminator Training 
        optimizer_D.zero_grad() 

        # discriminator Loss Calculate 
        realLoss = adversarial_loss(discriminator(real_imgs),
                                    real)
        fakeLoss = adversarial_loss(discriminator(generatedImg.detach()),
                                    fake) # generator not training 
        dLoss = (realLoss+fakeLoss) / 2
        dLoss.backward()
        optimizer_D.step() 

        done = epoch * len(dataloader) + i
        if done % sample_interval == 0:
            save_image(generatedImg.data[:25],f"{done}.png",nrow = 5,normalize = True)

    print(f"[Epoch {epoch}/{EPOCH}] [D loss:{dLoss.item():.6f}] [G loss: {gLoss.item():.6f}] [Elapsed time: {time.time() - start_time:.2f}s")

