import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

IMG_SIZE = 50

DATADIR = "E:/Pycharm Projects/Datasets/PetImages"
CATEGORIES = ["Dog", "Cat"]

training_data = []

print("Loading Data.........0%")
for category in CATEGORIES:
    path = DATADIR+"/"+category
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(path+"/"+img, cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
        except Exception as e:
            pass
    print("Loading Data........50%")
print("Dataset Loaded successfully")

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape((-1, IMG_SIZE*IMG_SIZE, 1))
X = X/255.0
