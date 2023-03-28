#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import torchvision.datasets as datasets

data_dir = '/Datasets/Framework/ILSVRC12_256/'
# your dataset path

# Data loading code
traindir = os.path.join(data_dir, 'train')
train_dataset = datasets.ImageFolder(traindir, None)
classes = train_dataset.classes
print("the number of total classes: {}".format(len(classes)))

seed = 1993
np.random.seed(seed)
subset_num = 100
subset_classes = np.random.choice(classes, subset_num, replace=False)
print("the number of subset classes: {}".format(len(subset_classes)))
print(subset_classes)

with open('./subset_train.txt','r') as f,open('./subset_test.txt','r') as t:
    for file in f.readlines():
        data = file.split()
        if not os.path.exists('./sub_train/'+str(data[1])):
            os.makedirs('./sub_train/'+str(data[1]))
        os.system('cp '+str(data[0])+' ./sub_train/'+str(data[1]))

print("Hello World")