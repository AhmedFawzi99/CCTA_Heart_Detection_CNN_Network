#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"

if physical_devices:
    # Restrict Tensorflow to only use the first GPU
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
            
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')   
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(physical_devices), "Physical GPUs, ", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        #Visible devices must be set before GPUs have been initialized.
        print(e)


# In[25]:


from tensorflow.keras.models import Sequential
from keras import optimizers
from numpy import asarray
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense,MaxPool2D,Flatten
import pydicom
from pydicom.data import get_testdata_files
import os
import numpy as np
import os.path
import json
import random

from os import path
from matplotlib import pyplot, cm, patches


# In[26]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=16):
        self.x = np.load('X_train.npy', mmap_mode='r')
        self.y = np.load('Y_train.npy', mmap_mode='r')
        self.batch_size = batch_size
        
    def __getitem__(self, index):
        return self.x[index * self.batch_size: (index+1)*self.batch_size], self.y[index * self.batch_size: (index+1)*self.batch_size]
    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))


# In[27]:


# print("Training sample size: {}".format(trainPct))


models = Sequential()


# Keras model with two hidden layer with 10 neurons each 
models.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(512,512,1),activation='relu'))    # Input layer => input_shape should be explicitly designated
models.add(MaxPool2D(pool_size=(2,2)))

models.add(Conv2D(filters=128,kernel_size=(4,4),activation='relu'))    # Input layer => input_shape should be explicitly designated
models.add(MaxPool2D(pool_size=(2,2)))

models.add(Conv2D(filters=64,kernel_size=(4,4),activation='relu'))    # Input layer => input_shape should be explicitly designated
models.add(MaxPool2D(pool_size=(2,2)))

models.add(Conv2D(filters=128,kernel_size=(4,4),activation='relu'))    # Input layer => input_shape should be explicitly designated
models.add(MaxPool2D(pool_size=(2,2)))

models.add(Conv2D(filters=32,kernel_size=(4,4),activation='relu'))    # Input layer => input_shape should be explicitly designated
models.add(MaxPool2D(pool_size=(2,2)))

models.add(Flatten())
models.add(Dense(64,activation='relu'))
models.add(Dense(64,activation='relu'))
models.add(Dense(32,activation='relu'))
models.add(Dense(1))




models.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mse']) 
models.summary()


# In[28]:




generator = DataGenerator()
models.fit_generator(generator ,epochs = 10,verbose =10)

                # actual figure of metrics computed
    
class EvalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, batch_size=16):
        self.x = np.load('X_test.npy', mmap_mode='r')
        self.y = np.load('Y_test.npy', mmap_mode='r')
        self.batch_size = batch_size
        
    def __getitem__(self, index):
        return self.x[index * self.batch_size: (index+1)*self.batch_size], self.y[index * self.batch_size: (index+1)*self.batch_size]
    def __len__(self):
        return int(np.ceil(self.x.shape[0] / self.batch_size))

eval_generator = EvalDataGenerator()
results = models.evaluate_generator(eval_generator)
print("Training before results for model")
print(models.metrics_names)     # list of metric names the model is employing
print(results)                 # actual figure of metrics computed


# In[37]:


def analyse(path):
    
    # Get ref file
    RefDs = pydicom.read_file(path)

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), 1)

    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
    y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
    z = np.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'

    X_data = np.zeros(1*512*512,dtype="float32")
    # X_data = []
    X_data=np.reshape(X_data,(1,512,512))
    new_array = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

    ds = pydicom.read_file(path,force=True)
    # store the raw image data
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    idx=0
    new_array[:,:,idx] = ds.pixel_array
    X_data[idx] = new_array[:,:,idx]


    print(X_data.shape)
    X_data=np.reshape(X_data,(1,512,512,1))
    print(X_data.shape)
    return X_data


# In[38]:


X=analyse('/media/ccta/a3ee3238-d74c-4a4f-9c7f-afc98e547c81/Ahmed Fawzi/NEW/ITAC STUDIES/S101/S101-00139.dcm')


# In[39]:


y=models.predict_generator(X)


# In[41]:


y


# In[ ]:




