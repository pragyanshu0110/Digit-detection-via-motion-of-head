
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import numpy as np




######## ARGUMENT PARSING
parser = argparse.ArgumentParser()

parser.add_argument("-p","--shapepredictor",required=True,help="path of shape_predictor data")
args=parser.parse_args()

###### call the detector
detector=dlib.get_frontal_face_detector()
####load the shape predictor
predictor=dlib.shape_predictor(args.shapepredictor)


capture = cv2.VideoCapture(0)
while True:
    ret,frame=capture.read()
    frame=imutils.resize(frame,width=400)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects=detector(gray,0) # return (no. of faces ) for bounding box


    for rect in rects:
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)
        #print('shape 34',shape.shape)
        print(shape[34][0],',',shape[34][1])
       	

        #make a circle on point of nose
        cv2.circle(frame,(shape[34][0],shape[34][1]),1,(0,0,255),-1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):
        break

cv2.destroyAllWindows()
capture.release()

#=================================================================================


