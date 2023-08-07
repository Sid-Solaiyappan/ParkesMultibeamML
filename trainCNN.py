import os
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import tqdm
import tensorflow as tf
import shutil
plt.style.use('default')

import keras 

#Needed neural network layer types
from keras.models import Sequential
from keras.layers import Dense, Dropout, Normalization
from keras.preprocessing.image import ImageDataGenerator

#Graph loss vs accuracy over epochs
# !pip install livelossplot -q
# from livelossplot import PlotLossesKeras
from tensorflow.keras.callbacks import ModelCheckpoint

#Pretrained Networks
from keras.applications.resnet import ResNet50
# from keras.applications.vgg16 import VGG16
# from keras.applications.vgg19 import VGG19
# from keras.applications.resnet_v2 import ResNet50V2
# from keras.applications.resnet import ResNet101
# from keras.applications.resnet_v2 import ResNet101V2
# from keras.applications.inception_resnet_v2 import InceptionResNetV2
# from keras.applications.inception_v3 import InceptionV3
# from keras.applications.mobilenet import MobileNet
# from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.applications.mobilenet_v3 import MobileNetV3,MobileNetV3Large, MobileNetV3Small
# from keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201

from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

###### Required params #############

train = tf.keras.utils.image_dataset_from_directory('train', labels="inferred", label_mode="int",image_size=(96, 128,))
test = tf.keras.utils.image_dataset_from_directory('test', labels="inferred", label_mode="int",image_size=(96, 128,),shuffle=False)

model = 'resnet50'
ntrain_samples = '20' #in thousands
nepochs = 100
RUN_NAME = f'cnnRuns/{model}train{ntrain_samples}k{nepochs}e.keras' #CHANGE EACH TIME

####################################

strategy = tf.distribute.MirroredStrategy(
     cross_device_ops=tf.distribute.ReductionToOneDevice())

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

checkpoint = ModelCheckpoint('model_weight.h5',monitor='val_loss',
save_weights_only=True, mode='min', verbose=0)
callbacks = [checkpoint]

with strategy.scope():
    model = Sequential()
    model.add(Normalization())
    model.add(ResNet50(
        include_top=False,
        pooling='avg',
        ))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    print(model.layers)
    model.layers[0].trainable = False

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

model.fit(train, epochs=nepochs, verbose=1, validation_data=test,callbacks=callbacks)#class_weight={0: 1, 1: 5}, )

model.save(RUN_NAME)