
# coding: utf-8

# In[1]:


# CS 512: Assignment 4
# Fall 2018
# Herick Jayesh Asmani
# A20399752

import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint   
import os
import cv2
import sys
from cnn import Model
from cnn import Model_compile


# In[2]:


def get_image():
    # Obtain the image by entering the path
    cmd_line = input("Type the Path of your image file: ")
    filename = cmd_line
    print(filename)
    print(os.path.exists(filename))
    
    img = cv2.imread(filename, 1)
    
#     cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#     cv2.imshow('image',img)
#     cv2.waitKey(0)

    return img


# In[3]:


def image_processing(image):
    # Processing the image to make it suitable for classification
    resized_image = cv2.resize(image, (28, 28)) 
    ima = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(ima, (5,5), 0)
    # ret, thres = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)/255
    cv2.namedWindow('original_image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('binary_image', cv2.WINDOW_NORMAL)
    cv2.imshow('original_image', image)
    cv2.imshow('binary_image', thres)
    cv2.waitKey(0)
    
    return thres


# In[6]:


# C:\Users\Herick\Desktop\MS\CS512 Computer Vision\Assignments\Numbers\Sketches
def main():
    while (True):
        img = get_image()
        processed_img = image_processing(img)
        
        # Loading the model from the Custom CNN file
        model_1 = Model_compile()
        
        # Loading the best weights of the Custom CNN model
        model_1.load_weights('./models/model.weights.best.hdf5')
        
        # Reshaping image to make it suitable for classifiction 
        new_img = np.reshape(processed_img,[1,28,28,1])
        
        # Predicting even (1) or odd (0) given input image
        pred = model_1.predict_classes(new_img)
        print(pred)
        if pred == 1:
            print("The digit is even")
        else:
            print("The digit is odd")
        
        print("--------------------------------------------------------------------------")
        key = input("Enter q if you want to exit else enter any other key to continue. ")
        if key == 'q':
            break
        


# In[7]:


if __name__ == '__main__':
    main()

