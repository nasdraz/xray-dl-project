import os
import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

# Set this to the 'chest_xray' directory
data_dir = "DIRECTORY HERE"
test_dir = os.path.join(data_dir, 'test')

model = tf.keras.models.load_model('saved_model/densenet_model.h5')

test_datagen = ImageDataGenerator(rescale=1./255.)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                    target_size=(224, 224),
                                                    class_mode='binary')

print(model.evaluate(test_generator))