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

X, y = [], []

## Images rescaling
datagen = ImageDataGenerator(rescale=1.0/255.0)

#   Load images by resizing and shuffling randomly
train_dataset = datagen.flow_from_directory(WORKING_DIRECTORY, target_size=(150, 150),batch_size=6400, shuffle=True)

### Seperate Dataset from  Data Genrator
X, y = train_dataset.next()

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# Number of samples after train test split
print("Number of samples after splitting into Training, validation & test set\n")

print("Train     \t",sorted(Counter(np.argmax(y_train, axis=1)).items()))
print("Validation\t",sorted(Counter(np.argmax(y_val, axis=1)).items()))
print("Test      \t",sorted(Counter(np.argmax(y_test, axis=1)).items()))



print("Number of samples in each class:\t", sorted(Counter(np.argmax(y, axis=1)).items()))

#   class labels as per indices
print("Classes Names according to index:\t", train_dataset.class_indices)
X_train = X_train.mean(3)
X_val = X_val.mean(3)
X_test = X_test.mean(3)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape)
print("y_val shape:", y_val.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# Inint persistence imager
pimgr = PersistenceImager(pixel_size=0.01)
plt.rcParams.update(plt.rcParamsDefault)

# Persistence diagrams for 2 directions of 0-d homology on training data
print('Computing lower star of training data')
X_train_ls = [lower_star_img(img) for img in tqdm(X_train)]
print('Computing upper star of training data')
X_train_us = [lower_star_img(-img) for img in tqdm(X_train)]

#Test plot
plot_diagrams(X_train_ls[0], lifetime=True)
plt.show()

#Imaging pipeline for training
print('Initializing Persitence Imaging')
X_train_ls_clean = [img[:-1] for img in X_train_ls]
X_train_us_clean = [img[:-1] for img in X_train_us]
pimgr.fit(X_train_ls_clean, skew=False)
X_train_ls_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_train_ls_clean)]
pimgr.fit(X_train_us_clean, skew=False)
X_train_us_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_train_us_clean)]

#Test image
pimgr.plot_image(X_train_us_im[0])
plt.show()


print('Saving training data')
#Should document these dicts somewhere
np.savez('datadumper_train', X_train=X_train, X_train_ls = X_train_ls_im, X_train_us = X_train_us_im, y_train=y_train, class_indices = train_dataset.class_indices)


#0d homology for val
print('Computing lower star of validation data')
X_val_ls = [lower_star_img(img) for img in tqdm(X_val)]
print('Computing upper star of validation data')
X_val_us = [lower_star_img(-img) for img in tqdm(X_val)]

#Test plot
plot_diagrams(X_val_ls[0], lifetime=True)
plt.show()

#Imaging pipeline for val
print('Initializing Persitence Imaging')
X_val_ls_clean = [img[:-1] for img in X_val_ls]
X_val_us_clean = [img[:-1] for img in X_val_us]
pimgr.fit(X_val_ls_clean, skew=False)
X_val_ls_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_val_ls_clean)]
pimgr.fit(X_val_us_clean, skew=False)
X_val_us_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_val_us_clean)]

#Test image
pimgr.plot_image(X_val_us_im[0])
plt.show()

print('Saving validation data')
np.savez('datadumper_val', X_val=X_val, X_val_ls = X_val_ls_im, X_val_us = X_val_us_im, y_val=y_val)

#Test split as above
print('Computing lower star of testing data')
X_test_ls = [lower_star_img(img) for img in tqdm(X_test)]
print('Computing upper star of testing data')
X_test_us = [lower_star_img(-img) for img in tqdm(X_test)]

#Test plot
plot_diagrams(X_test_ls[0], lifetime=True)
plt.show()

print('Initializing Persitence Imaging')
X_test_ls_clean = [img[:-1] for img in X_test_ls]
X_test_us_clean = [img[:-1] for img in X_test_us]
pimgr.fit(X_test_ls_clean, skew=False)
X_test_ls_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_test_ls_clean)]
pimgr.fit(X_test_us_clean, skew=False)
X_test_us_im =  [pimgr.transform(img, skew =False) for img in tqdm(X_test_us_clean)]

pimgr.plot_image(X_test_us_im[0])
plt.show()
print('Saving testing data')
np.savez('datadumper_test',X_test=X_test, X_test_ls = X_test_ls_im, X_test_us = X_test_us_im, y_test=y_test)


print('Hope it worked')
