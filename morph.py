import cv2
import numpy as np

image=cv2.imread('output_img.jpg',0) # output_img.jpg
kernel=np.ones((7,7),np.uint8)

erosion=cv2.erode(image,kernel,iterations=1)
dilation=cv2.dilate(image,kernel,iterations=1)
opening=cv2.morphologyEx(image,cv2.MORPH_OPEN,kernel)
closing=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
#cv2.imshow('new_out_ero',erosion)
#cv2.imshow('new_out_dil',dilation)
#cv2.imshow('new_out_open-',opening)
#cv2.imshow('new_out_clos',closing)

#=================blurring========================
blurr_after_dil=cv2.GaussianBlur(dilation,(5,5),0)
blurr_after_clos=cv2.GaussianBlur(closing,(5,5),0)
#cv2.imshow('blurr_after_dil',blurr_after_dil)
#v2.imshow('blurr_after_clos',blurr_after_clos)

cv2.imwrite('test1.jpg',erosion) #saving an img test1.jpg

#cv2.waitKey(0)
cv2.destroyAllWindows()
