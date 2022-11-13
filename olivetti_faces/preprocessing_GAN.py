# To use this file, just import the data and target
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from unet import UNet
from discriminator import Discriminator
from baseline import FaceRecognizer
from preprocessing import *
attack_targets = torch.tensor([[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]) # the label of person to generate his image
faceRecognizer = FaceRecognizer()
faceRecognizer.load_state_dict(torch.load('best_model.pt'))
# init generator
generator = UNet(n_channels=1, n_classes=len(attack_targets))
generator.load_state_dict(torch.load('best_generator.pt'))
discriminator = Discriminator()

# data from https://www.kaggle.com/code/serkanpeldek/face-recognition-on-olivetti-dataset/notebook
data=np.load("./data/olivetti_faces.npy")
target=np.load("./data/olivetti_faces_target.npy")

avg=0.5470424729887565
std=0.17251527
# normalize data
data = (data-avg)/std

data_discriminator = torch.tensor(data[200:])
target_discriminator = np.ones(200)
    
def generate(target):
    gan_input = torch.normal(0, 1, size=(1,1,64, 64)) # noise
    x = generator.forward(gan_input)
    return x[0][target]

def generated_image(target):
    x = generate(target)
    tensor_to_picture(x.detach().numpy())
    
def print_confidence(target):
    x = generate(target)
    print(torch.softmax(faceRecognizer.forward(x.reshape(1,1,64,64)), -1))
    
# generate a dataset for discriminator using current generator
def discriminator_make_dataset():
    # first 200 real, last 200 fake
    generated_labels = torch.ones(400)
    generated_labels[200:] = 0
    # 20 images for each person
    generated_images = torch.zeros(200,64,64)
    for i in range(200):
        generated_images[i] = generate(i%10).detach()
    generated_images = torch.cat((data_discriminator, generated_images), dim=0)
    #shuffle the set
    idx = torch.randperm(400)
    generated_images = generated_images[idx]
    generated_labels = generated_labels[idx]
    
    return generated_images.reshape(400,1,1,64,64), generated_labels.reshape(400,1)

