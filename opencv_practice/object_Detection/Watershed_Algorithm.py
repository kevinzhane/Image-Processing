import cv2
import numpy as np
import matplotlib.pyplot as plt



def display(img,name,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img,cmap='gray')
    plt.title(str(name))
    plt.show()

def display2(img1,img2,cmap='gray'):
    fig = plt.figure(figsize=(12,10))
    ax1 = fig.add_subplot(211)
    ax1.imshow(img1,cmap='gray')
    ax2 = fig.add_subplot(212)
    ax2.imshow(img2,cmap='gray')
    plt.show()

sep_coins = cv2.imread('../opencv_practice/DATA/pennies.jpg')

#display (sep_coins,'original image')

###############################################################

# Median Blur --> Grayscale --> Binary Threshold --> Find Contours
# ------> It is awful method

# Median Blur
sep_blur = cv2.medianBlur(sep_coins,25)
#display(sep_blur,'Median Blur')

# Grayscale
gray_sep_coins = cv2.cvtColor(sep_blur,cv2.COLOR_BGR2GRAY)


# Binary Threshold
ret, sep_thresh = cv2.threshold(gray_sep_coins,160,255,cv2.THRESH_BINARY_INV)
#display(sep_thresh,'Sep Threshold')

# Find Contours
image,contours, hierarchy = cv2.findContours(sep_thresh.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

external_contours = np.zeros(sep_coins.shape)
internal_contours = np.zeros(sep_coins.shape)

#print (hierarchy)
for i in range (len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours,contours,i,255,-1)

#display(external_contours,'It is awful')

#################################################################
# Try the Watershed Algorithm for Six_pennis

img = cv2.imread('../opencv_practice/DATA/pennies.jpg')
img = cv2.medianBlur(img,35)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Try the Otsu's method of threshold method for Watershed Algorithm
# It better than normal threshold
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#display(thresh,'OTSU Threshold')

# Noise Removal (Optional)
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#display(opening,'Noise Removal')

#### We need to remove the connect of each coins
#### Try distance transform ---> Let highlight the image

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#display(dist_transform,'distance transform')

## Find the sure foreground & background
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
display(sure_fg,'Distance transform Threshold ---> Sure_fg')

sure_fg = np.uint8(sure_fg)
sure_bg = cv2.dilate(opening,kernel,iterations=3)

display2(opening,sure_bg)
display(sure_bg,'Dilating of BackGround ---> Sure_bg')

## Define the unknown region
unknown = cv2.subtract(sure_bg,sure_fg)
display(unknown,'unknown region')

## Final creat the label markers for watershed

# 1. Get the marker
ret, markers = cv2.connectedComponents(sure_fg)

# 2. Label the marker
markers = markers + 1
markers[unknown==255] = 0  

# 3. Apply the watershed algorithm
markers = cv2.watershed(img,markers)
display(markers,'Watershed Algorithm')

image,contours, hierarchy = cv2.findContours(markers.copy(),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)


#print (hierarchy)
for i in range (len(contours)):

    if hierarchy[0][i][3] == -1:
        cv2.drawContours(sep_coins,contours,i,(255,255,0),10)

display(sep_coins,'Draw the contour')

