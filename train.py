import random
import numpy as np
import tensorflow.python.keras.callbacks
from scipy import misc
import tensorflow as tf
from inceptionv2 import InceptionResNetV1
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_addons as tfa
size = (200, 200)
train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.1)
train_generator = train_datagen.flow_from_directory("./data/cropped", target_size=size, batch_size=32,
                                                    class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory("./data/cropped",  # same directory as training data
                                                         target_size=(192, 192), batch_size=32,
                                                         class_mode='categorical', subset='validation')
model = InceptionResNetV1((size[0], size[1], 3))
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss=tfa.losses.TripletHardLoss())
history=model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[tensorflow.keras.callbacks.ModelCheckpoint("./checkpoints")]
)