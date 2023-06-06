import os
import glob

TRAIN_DATA_PATH = './sketch_training/'
VALID_DATA_PATH = './sketch_valid/'
TEST_DATA_PATH = './sketch_testing/'

def train2valid(train):
    valid = train.replace(TRAIN_DATA_PATH, VALID_DATA_PATH)
    os.rename(image, valid)

def train2test(train):
    test = train.replace(TRAIN_DATA_PATH, TEST_DATA_PATH)
    os.rename(image, test)
    
#Function call
print('Start Spliting Testing...')
count = 0
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
# classes = ['butterfly']
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(TRAIN_DATA_PATH + label +'/*'):
        count += 1
        if count % 5 == 0:
            if count % 2 == 0:
                train2test(image)
            else:
                train2valid(image)
        if count % 1000 == 0:
            print(str(count / 25139 * 100) + '%')