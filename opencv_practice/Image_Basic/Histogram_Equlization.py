import numpy as np 
import cv2
import matplotlib.pyplot as plt 


dark_horse = cv2.imread('../opencv_practice/DATA/horse.jpg')
show_horse = cv2.cvtColor(dark_horse, cv2.COLOR_BGR2RGB)
#plt.imshow(show_horse)
#plt.show()

rainbow = cv2.imread('../opencv_practice/DATA/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
#plt.imshow(show_rainbow)
#plt.show()

blue_bricks = cv2.imread('../opencv_practice/DATA/bricks.jpg')
show_brick = cv2.cvtColor(blue_bricks, cv2.COLOR_BGR2RGB)
#plt.imshow(show_brick)
#plt.show()

## OPENCV BGR ---> "B" Channel histogram  
hist_value1 = cv2.calcHist([dark_horse],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.subplot(231)
plt.plot(hist_value1)
hist_value2 = cv2.calcHist([rainbow],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.subplot(232)
plt.plot(hist_value2)
hist_value3 = cv2.calcHist([blue_bricks],channels=[0],mask=None,histSize=[256],ranges=[0,256])
plt.subplot(233)
plt.plot(hist_value3)

plt.subplot(234)
plt.imshow(show_horse)
plt.subplot(235)
plt.imshow(show_rainbow)
plt.subplot(236)
plt.imshow(show_brick)
plt.show()


## All Channel for histogram

color = {'b','g','r'}
img = blue_bricks
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
plt.title('Histrgram For Blue_Bricks')
plt.show()

img = dark_horse
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,50])
    plt.ylim([0,50000])
plt.title('Histrgram For Dark_horse')
plt.show()

img = rainbow
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color=col)
    plt.xlim([0,256])
    #plt.ylim([0,50000])
plt.title('Histrgram For rainbow')
plt.show()


## Histogram CALACULATE
img = rainbow
mask = np.zeros(img.shape[:2],np.uint8)
mask[300:400,100:400] = 255 #Build the mask range y axis & x axis
#plt.imshow(mask,cmap='gray')
#plt.show()

# For calculate histogram
masked_img = cv2.bitwise_and(img,img,mask=mask)
# For vistulization
show_masked_img = cv2.bitwise_and(show_rainbow,show_rainbow,mask=mask)

plt.subplot(211)
plt.imshow(show_rainbow)
plt.subplot(212)
plt.imshow(show_masked_img)
plt.show()


# B G R 
hist_mask_value_red = cv2.calcHist([rainbow],channels=[2],mask=mask,histSize=[256],ranges=[0,256])
hist_value_red = cv2.calcHist([rainbow],channels=[2],mask=None,histSize=[256],ranges=[0,256])
plt.subplot(121)
plt.plot(hist_mask_value_red)
plt.title('RED HISTOGRAM FOR MASKED RAINBOW')
plt.subplot(122)
plt.plot(hist_value_red)
plt.title('RED HISTOGRAM FOR NO MASKED RAINBOW')
plt.show()





# Histogram Equlization for gray scale
gorilla = cv2.imread('../opencv_practice/DATA/gorilla.jpg',0)
#plt.imshow(gorilla,cmap='gray')
#plt.show()

hist_values = cv2.calcHist([gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])
#plt.plot(hist_values)
#plt.show()

eq_gorilla = cv2.equalizeHist(gorilla)
eq_hist_value = cv2.calcHist([eq_gorilla],channels=[0],mask=None,histSize=[256],ranges=[0,256])

plt.subplot(221)
plt.title('Orgin')
plt.imshow(gorilla,cmap='gray')
plt.subplot(222)
plt.title('Equlization')
plt.imshow(eq_gorilla,cmap='gray')
plt.subplot(223)
plt.plot(hist_values)
plt.subplot(224)
plt.plot(eq_hist_value)
plt.show()



# Histogram Equlization for full color
color_gorilla = cv2.imread('../opencv_practice/DATA/gorilla.jpg')
show_gorilla = cv2.cvtColor(color_gorilla, cv2.COLOR_BGR2RGB)

hsv = cv2.cvtColor(color_gorilla,cv2.COLOR_BGR2HSV) # BGR--->HSV
hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2]) # USE Equilaztion value to instead orgin value
eq_color_gorilla = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB) # HSV--->RGB
plt.imshow(eq_color_gorilla)
plt.title('Full Color Image Histogram Equlization')
plt.show()