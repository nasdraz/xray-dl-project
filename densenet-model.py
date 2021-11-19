from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.applications.densenet import DenseNet201
from tensorflow.python.keras.models import Model
import matplotlib.pyplot as plt
import os

# Set this to the 'chest_xray' directory
data_dir = "DIRECTORY HERE"
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'val')


pre_trained_model = DenseNet201(input_shape=(224, 224, 3),
                                include_top=False,
                                weights='imagenet')

for layer in pre_trained_model.layers:
    layer.trainable = False

x = layers.Flatten()(pre_trained_model.output)
x = layers.Dense(2048, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['AUC', 'acc'])



train_datagen = ImageDataGenerator(rescale=1. / 255., rotation_range= 10, fill_mode="nearest")
validation_datagen = ImageDataGenerator(rescale=1./255.)



train_generator = train_datagen.flow_from_directory(train_dir,
                                                    shuffle=True,
                                                    target_size=(224, 224),
                                                    batch_size=20,
                                                    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                    target_size=(224, 224),
                                                    batch_size=4,
                                                    class_mode='binary')

history = model.fit(train_generator,
                              steps_per_epoch=390,
                              epochs=10,
                              validation_data = validation_generator,
                              validation_steps = 4,
                              verbose=2)

acc=history.history['acc']
val_acc = history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

model.save('saved_model/densenet_model.h5')

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label= 'training acc')
plt.plot(epochs, val_acc, 'b', label= 'validation acc')

plt.figure()

plt.plot(epochs, loss, 'r', label= 'training loss')
plt.plot(epochs, val_loss, 'b', label= 'validation loss')
plt.legend()

plt.show()

