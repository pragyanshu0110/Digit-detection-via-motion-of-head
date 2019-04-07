import os

#=============PART:1: Taking input image from camera =================
os.system(" python3 camera_input.py")

#==============PART:2: Doing morphological operations on image ================
os.system(" python3 morph.py")


#==============PART:3: Predicting the digit using trained model================
os.system(" python3 digit_prediction.py")
