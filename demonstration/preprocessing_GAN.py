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
# data from https://www.kaggle.com/code/serkanpeldek/face-recognition-on-olivetti-dataset/notebook
data=np.load("./data/olivetti_faces.npy")
target=np.load("./data/olivetti_faces_target.npy")

avg=0.5470424729887565
std=0.17251527
# normalize data
data = (data-avg)/std

    
def generate(generator, target):
    gan_input = torch.normal(0, 1, size=(1,1,64, 64)) # noise
    x = generator.forward(gan_input)
    return x[0][target]

def generated_image(generator, target):
    x = generate(generator, target)
    tensor_to_picture(x.detach().numpy())
    
def print_confidence(target):
    x = generate(target)
    print(torch.softmax(faceRecognizer.forward(x.reshape(1,1,64,64)), -1))
    


