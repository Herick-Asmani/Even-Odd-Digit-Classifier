
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
import matplotlib.cm as cm
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras import backend as K
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard   


# In[2]:


# Using Keras to import pre-shuffled MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


def even_odd_map(label):
    new_label = np.zeros((label.shape))
    for i in range(len(label)):
        if label[i] % 2 == 0:
            new_label[i] = 1        # Mapping even numbers to 1
        else:
            new_label[i] = 0        # Mapping odd numbers to 0
    
    return new_label 


# In[4]:


#def Pre_Processing(X_train, y_train, X_test, y_test):
#    new_Y_train = even_odd_map(y_train)
#    new_Y_test = even_odd_map(y_test)
#    # rescale [0,255] --> [0,1]
#    X_train = X_train.astype('float32')/255
#    X_test = X_test.astype('float32')/255
    
#    return X_train, new_Y_train, X_test, new_Y_test

# Mapping data labels (0-9) to Odd-even labels (0-1)
new_Y_train = even_odd_map(y_train)
new_Y_test = even_odd_map(y_test)

# print(y_train[:20])
# print(new_Y_train[:20])
# print(y_test[:20])
# print(new_Y_test[:20])

# Rescaling [0,255] --> [0,1]
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# print(X_train.shape[1:])

if keras.backend.image_data_format == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_train.shape[1], X_train.shape[2])
    ip_shape = (1, X_train.shape[1], X_train.shape[2])
else:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], X_train.shape[2], 1)
    ip_shape = (X_train.shape[1], X_train.shape[2], 1)   
    
# # one-hot encode the labels for softmax classifier
num_classes = len(np.unique(new_Y_train))
new_Y_train = keras.utils.to_categorical(new_Y_train, num_classes)
new_Y_test = keras.utils.to_categorical(new_Y_test, num_classes)

# print(num_classes)
# print(new_Y_train.shape)
# print(new_Y_test.shape)


# In[5]:


def Model():
# Creating the CNN model

    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', activation = 'relu', input_shape = ip_shape))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    #model.add(Dropout(0.4))
    model.add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation = 'softmax'))
    
    model.summary()
    return model


# In[6]:


def as_keras_metric(method):
    # Creating a function to calculate precision and recall
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


# In[7]:


def Model_compile():
    # Compiling the model
    
    model = Model()
    precision = as_keras_metric(tf.metrics.precision)
    recall = as_keras_metric(tf.metrics.recall)
    optim = keras.optimizers.SGD(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy', precision, recall])
    
    return model


# In[8]:


def main():
    model_1 = Model_compile()
    
    # Creating checkpoints to obtain better model weights and to obtain plots.
    checkpointer = ModelCheckpoint(filepath = './models/model.weights.best.hdf5', verbose = 1, save_best_only = True)
    tensorboard = TensorBoard(log_dir='./models', histogram_freq=0, write_graph=True, write_images=True)
    callback = [checkpointer, tensorboard]
    
    # Fitting the model
    m_fit = model_1.fit(X_train, new_Y_train, epochs = 5, verbose = 1, validation_data=(X_test, new_Y_test), callbacks = callback)
    
    # loading the weights that yielded the best validation loss
    model_1.load_weights('./models/model.weights.best.hdf5')

    # evaluate and print test loss, accuracy, precision and recall
    score = model_1.evaluate(X_test, new_Y_test, verbose=0)
    
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Test precision: ', score[2])
    print('Test recall: ', score[3])


# In[9]:


if __name__ == '__main__':
    main()


# In[88]:




