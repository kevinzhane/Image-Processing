import cv2
import matplotlib.pyplot as plt
import numpy as np


def display(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


img = cv2.imread('../opencv_practice/DATA/car_plate.jpg')
#display(img)

# Read cascade .xml
car_plate_cascade = cv2.CascadeClassifier('../opencv_practice/DATA/haarcascades/haarcascade_russian_plate_number.xml')

# Build detect model
def detect_plate(img):

    plate_img = img.copy()
    plate_rects = car_plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2,minNeighbors=5)
    #print (plate_rects)

    for (x,y,w,h) in plate_rects:
        cv2.rectangle(plate_img,(x,y),(x+w,y+h),(0,0,255),5)

    return plate_img
results = detect_plate(img)
display(results)

def detect_and_blur_plate(img):


    plate_img = img.copy()
    roi = img.copy()
    plate_rects = car_plate_cascade.detectMultiScale(plate_img,scaleFactor=1.2,minNeighbors=5)
    #print (plate_rects)

    for (x,y,w,h) in plate_rects:
        
        # Build the ROI and blur it
        roi = roi[y:y+h,x:x+w]
        blurred_roi = cv2.medianBlur(roi,11)
        plate_img[y:y+h,x:x+w] = blurred_roi 

    return plate_img

results = detect_and_blur_plate(img)
display(results)

