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
from persim.images_weights import linear_ramp

# Create EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


## Set Path Here before running the code
WORKING_DIRECTORY =  os.getcwd()
print(os.getcwd())
##  Name of classes
#CLASSES = ['Mild_Demented','Moderate_Demented','Non_Demented','Very_Mild_Demented']

X, y = [], []

## Images 
datagen = ImageDataGenerator()

#   Load images by resizing and shuffling randomly
train_dataset = datagen.flow_from_directory(WORKING_DIRECTORY, target_size=(150, 150),batch_size=6400, shuffle=True)

### Seperate Dataset from  Data Genrator
X, y = train_dataset.next()
X = X.mean(3)

# Persistence diagrams for 0-d homology
print('Computing lower star')
X_ls = [lower_star_img(img) for img in tqdm(X)]

plt.rcParams.update(plt.rcParamsDefault)
#plot_diagrams(X_ls[0])
#plt.show()

#Imaging pipelineg
print('Initializing Persitence Imaging')
X_clean = [img[:-1] for img in X_ls]

pimgr = PersistenceImager(pixel_size=25)

pimgr.fit(X_clean, skew=False)
X_ls_im =  np.array([pimgr.transform(img, skew =True) for img in tqdm(X_clean)])
print(X_ls_im[0].shape)
pimgr.plot_image(X_ls_im[0])
plt.show()

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_ls_im, y, test_size=0.2, random_state=42)



# Number of samples after train test split
print("Number of samples after splitting into Training & test set\n")

print("Train     \t",sorted(Counter(np.argmax(y_train, axis=1)).items()))
#print("Validation\t",sorted(Counter(np.argmax(y_val, axis=1)).items()))
print("Test      \t",sorted(Counter(np.argmax(y_test, axis=1)).items()))



print("Number of samples in each class:\t", sorted(Counter(np.argmax(y, axis=1)).items()))

#   class labels as per indices
print("Classes Names according to index:\t", train_dataset.class_indices)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
#print("X_val shape:", X_val.shape)
#print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)



print('Saving  data')
#Should document these dicts somewhere
np.savez('datadumper', X_train=X_train, X_test = X_test,  y_train=y_train, y_test=y_test, class_indices = train_dataset.class_indices)
print('Hope it worked')
exit()

