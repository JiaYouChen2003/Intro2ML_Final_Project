import numpy as np
import os
import glob
from PIL import Image
import torch
import random
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt

IMG_SIZE_X = 64
IMG_SIZE_Y = 64
# 0 2010 4020 6030 10050 20112
EXTRA_SKETCH_TRAIN_IMG_COUNT = 2010
# 25139 * 0.8
TRAIN_IMG_COUNT = 20112 + EXTRA_SKETCH_TRAIN_IMG_COUNT
# 25139 * 0.1
VALID_IMG_COUNT = 2514
# 25139 * 0.1
TEST_IMG_COUNT = 2513

LR = 0.001
BATCH_SIZE = 128
EPOCH = 10000

TRAIN_DATA_PATH = './training/'
VALID_DATA_PATH = './valid/'
TEST_DATA_PATH = './testing/'
SKETCH_TRAIN_DATA_PATH = './sketch_training/'
SKETCH_VALID_DATA_PATH = './sketch_valid/'
SKETCH_TEST_DATA_PATH = './sketch_testing/'

MODELS_PATH = './models'
MODEL_NAME = 'VGG19'

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        
        self.fc0 = nn.MaxPool2d(kernel_size = 1, stride = 1)

        # in[N, 512] => out[N, 10]
        self.out = nn.Linear(512, 10)

    def forward(self, x):
        # x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.fc0(x)
        x = x.view(x.size(0), -1) # [N, 32 * 8 * 8]
        output = self.out(x)
        return output

train_imgs = np.ones((TRAIN_IMG_COUNT, 3, IMG_SIZE_X, IMG_SIZE_Y), dtype = int)
train_label = np.zeros((TRAIN_IMG_COUNT), dtype = int)
valid_imgs = np.zeros((VALID_IMG_COUNT, 3, IMG_SIZE_X, IMG_SIZE_Y), dtype = int)
valid_label = np.zeros((VALID_IMG_COUNT), dtype = int)
sketch_valid_imgs = np.zeros((VALID_IMG_COUNT, 3, IMG_SIZE_X, IMG_SIZE_Y), dtype = int)
sketch_valid_label = np.zeros((VALID_IMG_COUNT), dtype = int)
test_imgs = np.zeros((TEST_IMG_COUNT, 3, IMG_SIZE_X, IMG_SIZE_Y), dtype = int)
test_label = np.zeros((TEST_IMG_COUNT), dtype = int)

classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
# classes = ['butterfly']

count = 0
label_count = 0
print('Start Load Training Data')
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(TRAIN_DATA_PATH + label +'/*'): # sheep
        img = np.asarray(Image.open(image).convert('RGB').resize((IMG_SIZE_X, IMG_SIZE_Y)), dtype = float)
        img = np.expand_dims(img, axis = 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        
        train_imgs[count, :] = img
        train_label[count] = i
        count += 1
        if (count % 1000 == 0):
            print(str(round(float(count / TRAIN_IMG_COUNT * 100), 2)) + '%')
    
    for image in glob.glob(SKETCH_TRAIN_DATA_PATH + label +'/*'): # sheep
        img = np.asarray(Image.open(image).convert('RGB').resize((IMG_SIZE_X, IMG_SIZE_Y)), dtype = float)
        img = np.expand_dims(img, axis = 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        
        train_imgs[count, :] = img
        train_label[count] = i
        count += 1
        if (count % 1000 == 0):
            print(str(round(float(count / TRAIN_IMG_COUNT * 100), 2)) + '%')
        label_count += 1
        if (label_count >= i * EXTRA_SKETCH_TRAIN_IMG_COUNT / 10):
            print('Finish load ' + label)
            break
print(str(round(float(count / TRAIN_IMG_COUNT * 100), 2)) + '%')

count = 0
print('Start Load Valid Data')
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(VALID_DATA_PATH + label +'/*'): # sheep
        img = np.asarray(Image.open(image).convert('RGB').resize((IMG_SIZE_X, IMG_SIZE_Y)), dtype = float)
        img = np.expand_dims(img, axis = 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        
        valid_imgs[count, :] = img
        valid_label[count] = i
        count += 1
        if (count % 1000 == 0):
            print(str(round(float(count / VALID_IMG_COUNT * 100), 2)) + '%')
print(str(round(float(count / VALID_IMG_COUNT * 100), 2)) + '%')

count = 0
print('Start Load Sketch Valid Data')
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(SKETCH_VALID_DATA_PATH + label +'/*'): # sheep
        img = np.asarray(Image.open(image).convert('RGB').resize((IMG_SIZE_X, IMG_SIZE_Y)), dtype = float)
        img = np.expand_dims(img, axis = 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        
        sketch_valid_imgs[count, :] = img
        sketch_valid_label[count] = i
        count += 1
        if (count % 1000 == 0):
            print(str(round(float(count / VALID_IMG_COUNT * 100), 2)) + '%')
print(str(round(float(count / VALID_IMG_COUNT * 100), 2)) + '%')

count = 0
print('Start Load Testing Data')
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(TEST_DATA_PATH + label +'/*'): # sheep
        img = np.asarray(Image.open(image).convert('RGB').resize((IMG_SIZE_X, IMG_SIZE_Y)), dtype = float)
        img = np.expand_dims(img, axis = 0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = torch.from_numpy(img)
        
        test_imgs[count, :] = img
        test_label[count] = i
        count += 1
        if (count % 1000 == 0):
            print(str(round(float(count / TEST_IMG_COUNT * 100), 2)) + '%')
print(str(round(float(count / TEST_IMG_COUNT * 100), 2)) + '%')

print('Finish Load Data')

device = 0
if (torch.cuda.is_available()):
    device = 'cuda'
train_imgs = torch.from_numpy(train_imgs)
train_imgs = torch.tensor(train_imgs, dtype = torch.float)
train_label = torch.from_numpy(train_label)

valid_imgs = torch.from_numpy(valid_imgs)
valid_imgs = torch.tensor(valid_imgs, dtype = torch.float)
valid_label = torch.from_numpy(valid_label)

sketch_valid_imgs = torch.from_numpy(sketch_valid_imgs)
sketch_valid_imgs = torch.tensor(sketch_valid_imgs, dtype = torch.float)
sketch_valid_label = torch.from_numpy(sketch_valid_label)

test_imgs = torch.from_numpy(test_imgs)
test_imgs = torch.tensor(test_imgs, dtype = torch.float)
test_label = torch.from_numpy(test_label)


class TrainingDataset():
    def __init__(self):
        self.x = train_imgs
        self.y = train_label
        self.n_samples = train_imgs.shape[0]

    def __getitem__(self, index):
        x = self.x[index, :]
        y = self.y[index]
        # if random.random() > 0.9:
        #     x[0, :] = x[0, :] * 299/1000 + x[1, :] * 587/1000 + x[2, :] * 114/1000
        #     x[1, :] = x[0, :]
        #     x[2, :] = x[0, :]
        # if random.random() > 0.7:
        #     x = torch.flip(x, [1])
        if random.random() > 0.5:
            x = torch.flip(x, [2])
        return x, y
    
    def __len__(self):
        return self.n_samples
train_data_loader = Data.DataLoader(dataset = TrainingDataset(), batch_size = BATCH_SIZE, shuffle = True)

class ValidDataset():
    def __init__(self):
        self.x = valid_imgs
        self.y = valid_label
        self.n_samples = valid_imgs.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]
    
    def __len__(self):
        return self.n_samples
valid_data_loader = Data.DataLoader(dataset = ValidDataset(), batch_size = BATCH_SIZE, shuffle = True)

class SketchValidDataset():
    def __init__(self):
        self.x = sketch_valid_imgs
        self.y = sketch_valid_label
        self.n_samples = sketch_valid_imgs.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]
    
    def __len__(self):
        return self.n_samples
sketch_valid_data_loader = Data.DataLoader(dataset = SketchValidDataset(), batch_size = BATCH_SIZE, shuffle = True)

class TestingDataset():
    def __init__(self):
        self.x = test_imgs
        self.y = test_label
        self.n_samples = test_imgs.shape[0]

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]
    
    def __len__(self):
        return self.n_samples
test_data_loader = Data.DataLoader(dataset = TestingDataset(), batch_size = BATCH_SIZE, shuffle = True)

model = CNN().to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)
if not os.path.exists(MODELS_PATH):
    os.mkdir(MODELS_PATH)

print('Start Training')
with open('valid_sketch_acc.txt', 'a') as valid_sketch_acc_file, open('valid_acc.txt', 'a') as valid_acc_file, open('train_loss.txt', 'a') as train_loss_file:
    for epoch in range(EPOCH):
        model.train()
        step = 0
        loss = 0
        for step, (x, y) in enumerate(train_data_loader):
            out = model(x.to(device))
            loss = loss_function(out, y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print('Epoch: {} | Loss: {}'.format(epoch + 1, loss))
        train_loss_file.write('Epoch: {} | Loss: {}'.format(epoch + 1, loss))
        
        if epoch % 10 == 0:
            torch.save(model, os.path.join(MODELS_PATH, MODEL_NAME + '_Epoch' + str(epoch + 1) + '.pt'))
            model.eval()
            for step, (x, y) in enumerate(valid_data_loader):
                prediction = torch.argmax(model(x.to(device)), 1)
                if step == 0:
                    acc = torch.eq(prediction, y.to(device))
                else:
                    acc = torch.cat((acc, torch.eq(prediction, y.to(device))), 0)
            acc = (torch.sum(acc) / acc.shape[0]).item()
            print('Accuracy: {:.2%}'.format(acc))
            valid_acc_file.write('Epoch: {} Acc: {:.2%}\n'.format(epoch + 1, acc))
            
            for step, (x, y) in enumerate(sketch_valid_data_loader):
                prediction = torch.argmax(model(x.to(device)), 1)
                if step == 0:
                    acc = torch.eq(prediction, y.to(device))
                else:
                    acc = torch.cat((acc, torch.eq(prediction, y.to(device))), 0)
            acc = (torch.sum(acc) / acc.shape[0]).item()
            print('Sketch Accuracy: {:.2%}'.format(acc))
            valid_sketch_acc_file.write('Epoch: {} Sketch Acc: {:.2%}\n'.format(epoch + 1, acc))