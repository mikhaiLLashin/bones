import tensorflow as tf
from keras.layers import Input,Dense,Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19,preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import glob
from matplotlib import pyplot as plt
import warnings
from datetime import datetime
from keras.callbacks import ModelCheckpoint

IMAGE_SIZE = [ 800 , 500 , 3 ]
vgg = VGG19( include_top = False,
            input_shape = IMAGE_SIZE,
            weights = 'imagenet')
for  layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense( 9 , activation = 'softmax' )(x)

model = Model( inputs = vgg.input , outputs = prediction )

train_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    rotation_range = 40 ,
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 ,
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True ,
    fill_mode = 'nearest'
)
test_datagen = ImageDataGenerator(
    preprocessing_function = preprocess_input ,
    rotation_range = 40 ,
    width_shift_range = 0.2 ,
    height_shift_range = 0.2 ,
    shear_range = 0.2 ,
    zoom_range = 0.2 ,
    horizontal_flip = True ,
    fill_mode = 'nearest'
)

adam=Adam()

model.compile( loss = 'categorical_crossentropy',
              optimizer = adam,
              metrics = ['accuracy'] )

train_path = "crop_imgs_train/big"
test_path = "crop_imgs_test/big"


train_set = train_datagen.flow_from_directory(train_path,
                                            target_size = ( 800 , 500 ),
                                            batch_size = 5,
                                            class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                             target_size = ( 800 , 500 ),
                                            batch_size = 1,
                                            class_mode = 'categorical')


'''checkpoint = ModelCheckpoint(filepath = 'version1.h5' , verbose = 2 , save_best_only = True )
callbacks = [checkpoint]
start = datetime.now()
model_history = model.fit( train_set,
                          validation_data = test_set,
                          epochs = 500,
                          steps_per_epoch = 5,
                          validation_steps = 32,
                          callbacks = callbacks,
                          verbose = 2)'''

model.load_weights("version1.h5")

for i in model.predict(test_set):
    m = 0
    ma = 0
    for j in range(len(i)):
        if (m < i[j]):
            m = i[j]
            ma = j
    print(ma)
print(model.predict(test_set))