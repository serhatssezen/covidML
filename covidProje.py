# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B4_x330epABwa2iDExpv1W_iFRQiwbsh
"""

import pandas as pd
import os 
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers import Activation, Dropout, Flatten, Dense

"""# **DATA SETİMİZİ GOOGLE DRİVE İLE YÜKLÜYORUZ**"""

from google.colab import drive
drive.mount("/content/gdrive")

os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/MyDrive"

"/content/gdrive/MyDrive"

!pwd

"""# **ZİP DOSYASINDAN ÇIKARTARAK DATAMIZA ULAŞIYORUZ**"""

!kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --force

!ls

!unzip

!unzip \*.zip && *.zip

tf.test.gpu_device_name()
device_lib.list_local_devices()

img_width, img_height = 224, 224

"""# **EĞİTİM ESNASINDA KAÇ DEFA ÖĞRETECEĞİMİZİ VE BATCH'İMİZİ VERDİK**"""

train_data_yolu = "/content/chest_xray/train"
validation_data_yolu = "/content/chest_xray/test"
train_ornek_sayisi = 5216
validation_ornek_sayisi= 624
epochs = 5
batch_size = 16

model = Sequential()

model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (5,5)))
model.add(Conv2D(32, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))

model.add(Activation("sigmoid"))

tf.config.run_functions_eagerly(True)

tf.config.experimental_run_functions_eagerly(True)

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

train_datalar = ImageDataGenerator(
    rescale=1. /255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datalar = ImageDataGenerator(rescale=1./255)

train_generator = train_datalar.flow_from_directory(
    train_data_yolu,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical"
)
validation_generator = test_datalar.flow_from_directory(
    validation_data_yolu,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical"
)

model.fit_generator(
    train_generator,
    steps_per_epoch=train_ornek_sayisi // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_ornek_sayisi // batch_size
)