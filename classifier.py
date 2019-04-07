# Import the modules
import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")
#print(dataset)

# Extract the features and labels
features = np.array(dataset.data, 'int16') 
labels = np.array(dataset.target, 'int')

#print(labels)

# Create an linear SVM object
clsf = LinearSVC()

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Perform the training
clsf.fit(hog_features,labels)

# Save the classifier
joblib.dump(clsf, "pragya.pkl",compress=3)  # compress for save space on disk
