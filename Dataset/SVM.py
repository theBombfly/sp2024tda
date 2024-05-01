import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from persim import plot_diagrams, PersistenceImager
from ripser import ripser, lower_star_img
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

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

print(dic)
#exit()
Xt = dic['X_train']
Xt = np.array([img.flatten() for img in Xt])
Xte = dic['X_test']
Xte = np.array([img.flatten() for img in Xte])
yt = np.argmax(dic['y_train'], axis=1)
yte = np.argmax(dic['y_test'], axis=1)


print(Xt)
print(Xt.shape)
print(yt.shape)

clf = SVC(gamma='auto', verbose=True, decision_function_shape='ovo')
clf.fit(Xt, yt)
print(clf.score(Xt, yt))
print(clf.score(Xte, yte))