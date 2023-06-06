import os
import glob

TRAIN_DATA_PATH = './sketch_training/'
VALID_DATA_PATH = './sketch_valid/'
TEST_DATA_PATH = './sketch_testing/'
    
#Function call
print('Start Counting Training...')
count = 0
classes = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spyder', 'squirrel']
# classes = ['butterfly']
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(TRAIN_DATA_PATH + label +'/*'):
        count += 1
print('Training data:' + str(count))

print('Start Counting Valid...')
count = 0
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(VALID_DATA_PATH + label +'/*'):
        count += 1
print('Valid data:' + str(count))

print('Start Counting Testing...')
count = 0
for i in range(len(classes)):
    label = classes[i]
    for image in glob.glob(TEST_DATA_PATH + label +'/*'):
        count += 1
print('Testing data:' + str(count))