import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from persim import plot_diagrams, PersistenceImager
from ripser import ripser, lower_star_img
from tqdm import tqdm

# Create EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


## Set Path Here before running the code
WORKING_DIRECTORY =  os.getcwd()
print(os.getcwd())
##  Name of classes
CLASSES = ['Mild_Demented',
           'Moderate_Demented',
           'Non_Demented',
           'Very_Mild_Demented']

dic = np.load('datadumper.npz')
X_train = dic['X_train']
X_test = dic['X_test']
y_train = np.argmax(dic['y_train'], axis=1).reshape(-1,1)
y_test = np.argmax(dic['y_test'], axis=1).reshape(-1,1)


print(X_train.shape)


#exit()
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(25,25,1), padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, 
                    validation_split=0.2)
plt.rcParams.update(plt.rcParamsDefault)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(test_acc)
