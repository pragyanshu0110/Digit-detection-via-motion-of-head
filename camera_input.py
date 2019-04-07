import numpy as np
import cv2
import os

#=============PART:1:detect the points ==========================================

os.system(" python3 headmean.py \--shapepredictor shape_predictor_68_face_landmarks.dat > abc.txt")

#=============PART:2:converting array into image ================================ 
norm_array=np.ones((500,500))
norm_array=np.multiply(255,norm_array)
data=np.loadtxt('abc.txt',delimiter=',')


for point in data:
    norm_array[int(point[1]),int(point[0])]=0
	
			
#print(norm_array)

img=np.asarray(norm_array) # convert array to img

cv2.imwrite('output_img.jpg',img) #saving an img

#=================================================================

image=cv2.imread('output_img.jpg',0)
kernel=np.ones((5,5),np.uint8)

erosion=cv2.erode(image,kernel,iterations=1)
dilation=cv2.dilate(image,kernel,iterations=1)
opening=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
#cv2.imshow('new_out_dil',dilation)
#cv2.imshow('new_out_clos',closing)
cv2.imwrite('test.jpg',closing) #saving an img
#cv2.waitKey(0)
cv2.destroyAllWindows()

#================= do the morphology part============
os.system(" python3 morph.py ")

