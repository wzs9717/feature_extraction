import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import sys
sys.setrecursionlimit(10000)

#import densenet
import numpy as np
import sklearn.metrics as metrics
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential,load_model
from PIL import Image
import tensorflow as tf
import gc
import pandas as pd
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D

from keras.utils import multi_gpu_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from scipy import io
import struct

def pic_transfer(X):
    # pic_tem2=X.resize((224, 224),Image.BILINEAR)
    X_resize=img_to_array(X)
    return X_resize

def main():
    model=load_model('./weights/MobilenetV2-CIFAR100.h5')

    layer_1 = K.function([model.layers[0].input], [model.get_layer('global_average_pooling2d_1').output])
    images=cv2.imread('./data/train/'+str(0)+'.jpg')
    data = np.expand_dims(images, axis=0)
    # data = pic_transfer(images)
    print(np.max(data[:]))
    # print(images.shape)
    f1 = layer_1([data/ 255.0])[0]
    print(f1.shape)
if __name__ == '__main__':
    main()