# To use this file, just import the data and target
import numpy as np
import matplotlib.pyplot as plt
# data from https://www.kaggle.com/code/serkanpeldek/face-recognition-on-olivetti-dataset/notebook
data=np.load("./data/olivetti_faces.npy")
target=np.load("./data/olivetti_faces_target.npy")
# print(data.shape)
# print(target.shape)
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
    if i%10 <= 2: # select 2 pictures for each person as training set
        test_data.append(data[i])
        test_target.append(target[i])
        data = np.delete(data, i, axis=0)
        target = np.delete(target, i)
        
tup = np.array(list(zip(data, target)),dtype=object)
np.random.seed(10)
np.random.shuffle(tup)
train_data, train_target = zip(*tup)
