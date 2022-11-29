# To use this file, just import the data and target
import numpy as np
import matplotlib.pyplot as plt
import torch
# data from https://www.kaggle.com/code/serkanpeldek/face-recognition-on-olivetti-dataset/notebook
data=np.load("./data/olivetti_faces.npy")
target=np.load("./data/olivetti_faces_target.npy")

avg=0.5470424729887565
std=0.17251527
# normalize data
data = (data-avg)/std

from collections import Counter
# Counter(target)
def show_picture_and_label(data, targets, i):
    f = plt.figure()
    plt.imshow(data[i],cmap='gray')
    plt.show()
    print("label is: ", targets[i])
    
test_data = []
test_target = []
for i in reversed(range(400)):
    if i%10 <= 2: # select 3 pictures for each person as training set
        test_data.append(data[i])
        test_target.append(target[i])
        data = np.delete(data, i, axis=0)
        target = np.delete(target, i)
        
tup = np.array(list(zip(data, target)),dtype=object)
np.random.seed(10)
np.random.shuffle(tup)
train_data, train_target = zip(*tup)

# train_data: tuple of length 320, each is a matrix represents a picture
# train_target: tuple of length 320, each is a label
# For training, there are 40 classes, each has 7 pictures
# For testing, there are 40 classes, each has 3 pictures

train_data = torch.tensor(train_data)
train_target = torch.tensor(train_target)
test_data = torch.tensor(test_data)
test_target = torch.tensor(test_target)

# Since the dataset is small, we assumed batch size = 1
train_data = train_data.reshape(280,1,1,64,64)
test_data = test_data.reshape(120,1,1,64,64)

train_target = train_target.reshape(len(train_target), 1)
test_target = test_target.reshape(len(test_target), 1)

# input tensor, output picture
def tensor_to_picture(data):
    f = plt.figure()
    plt.imshow(data,cmap='gray')
    plt.show()
    
# input target, output picture
def label_to_picture(target):
    f = plt.figure()
    for i in range(280):
        if train_target[i] == target:           
            plt.imshow(train_data[i][0][0],cmap='gray')
            plt.show()

# input a confusion matrix, print out the evaluation
def eval_confusion_matrix(mtx):
    correct = 0
    for i in range(mtx.shape[0]):
        for j in range(mtx.shape[1]):
            if i == j:
                correct += mtx[i][j]
    print("accuracy = ", correct/120)
                
    