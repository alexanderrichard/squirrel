#!/usr/bin/python2.7

import numpy as np
import cv2
import os

os.system('mkdir -p data/images')

# download and convert test images
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
os.system('gunzip t10k-images-idx3-ubyte.gz')
img_files = []
with open('t10k-images-idx3-ubyte', 'r') as f:
    f.read(16) # skip header
    data = np.fromfile(f, dtype=np.uint8).reshape((10000, 28, 28))
    for i in range(10000):
        img_files.append('data/images/test-' + str(i) + '.png')
        cv2.imwrite(img_files[-1], data[i,:,:])
os.remove('t10k-images-idx3-ubyte')
with open('data/test.images', 'w') as f:
    f.write('#images\n28 28 1\n')
    f.write('\n'.join(img_files) + '\n')

# download and convert train images
os.system('wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
os.system('gunzip train-images-idx3-ubyte.gz')
img_files = []
with open('train-images-idx3-ubyte', 'r') as f:
    f.read(16) # skip header
    data = np.fromfile(f, dtype=np.uint8).reshape((60000, 28, 28))
    for i in range(60000):
        img_files.append('data/images/train-' + str(i) + '.png')
        cv2.imwrite(img_files[-1], data[i,:,:])
os.remove('train-images-idx3-ubyte')
with open('data/train.images', 'w') as f:
    f.write('#images\n28 28 1\n')
    f.write('\n'.join(img_files) + '\n')

# download and convert test labels
os.system('wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
os.system('gunzip t10k-labels-idx1-ubyte.gz')
with open('t10k-labels-idx1-ubyte', 'r') as f:
    f.read(8) # skip header
    data = np.fromfile(f, dtype=np.uint8)
    np.savetxt('data/test.labels', data, fmt='%d', comments='', header='#labels\n10000 10')
os.remove('t10k-labels-idx1-ubyte')

# download and convert train labels
os.system('wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
os.system('gunzip train-labels-idx1-ubyte.gz')
with open('train-labels-idx1-ubyte', 'r') as f:
    f.read(8) # skip header
    data = np.fromfile(f, dtype=np.uint8)
    np.savetxt('data/train.labels', data, fmt='%d', comments='', header='#labels\n60000 10')
os.remove('train-labels-idx1-ubyte')


