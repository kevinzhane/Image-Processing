import numpy as np
import cv2
import matplotlib.pyplot as plt



img = cv2.imread('/home/kevin/Desktop/Image processing/Computer-Vision-with-Python/DATA/00-puppy.jpg')
print (type(img))
print (img.shape)
#plt.imshow(img)
#plt.show()

### matplotlib ---> R G B 
###	OPENCV ---> B G R

fix_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#plt.imshow(fix_img)
#plt.show()

img_gray = cv2.imread('/home/kevin/Desktop/Image processing/data/Computer-Vision-with-Python/DATA/00-puppy.jpg',cv2.IMREAD_GRAYSCALE)
print (img_gray.shape)
#plt.imshow(img_gray,cmap='gray')
#plt.show()

### Resize the image
new_img = cv2.resize(fix_img,(1000,400))
print (new_img.shape)
#plt.imshow(new_img)
#plt.show()

### 50% of origin image
w_ratio = 0.5
h_ratio = 0.5

new_img = cv2.resize(fix_img,(0,0),fix_img,w_ratio,h_ratio)
print (new_img.shape)
#plt.imshow(new_img)
#plt.show()


## flip
new_img = cv2.flip(fix_img,1)
#plt.imshow(new_img)
#plt.show()

print (cv2.imwrite('totally_new.jpg',fix_img))

fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
ax.imshow(fix_img)
plt.show()