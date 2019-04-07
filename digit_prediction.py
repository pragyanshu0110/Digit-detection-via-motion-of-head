from sklearn.externals import joblib
from skimage.feature import hog 
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the classifier
clsf = joblib.load("pragya.pkl")

# import image and resize it
img = cv2.imread("test1.jpg")

# Convert to grayscale and apply Gaussian filtering
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Threshold the image
ret, img_th = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #/////////////////////////////////////////////

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]  # it returns [x,y,w,h]

# For each rectangular region, calculate HOG features and predict
# the digit using Linear SVM.
#print(rects)
rect=rects[0]   # here only one contour is present
#print(rect)
# Make the rectangular region around the digit
leng = int(rect[3] * 1.6)
pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
roi = img_th[pt1:pt1+leng, pt2:pt2+leng]


# Resize the image
roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
roi = cv2.dilate(roi, (3, 3))
# Calculate the HOG features
roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#print('roihogfd',np.array([roi_hog_fd]))
nbr = clsf.predict(np.array([roi_hog_fd], 'float64'))
#cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
print(int(nbr[0]))


