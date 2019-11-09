import numpy as np 
import cv2
import matplotlib.pyplot as plt 

def load_img():
	blank_img = np.zeros((600,600))
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(blank_img,text='ABCDE',org=(50,300), fontFace=font, fontScale=5, color=(255,255,255),thickness=25)
	return blank_img

def display_img(img):
	fig = plt.figure(figsize=(12,10))
	ax = fig.add_subplot(111)
	ax.imshow(img,cmap='gray')
	plt.show()

def display_2img(img1,img2,str1,str2):
	
	fig = plt.figure(figsize=(12,12))
	ax1 = fig.add_subplot(221)
	ax1.imshow(img1,cmap='gray')
	plt.title(str1)
	ax2 = fig.add_subplot(222)
	ax2.imshow(img2,cmap='gray')
	plt.title(str2)
	plt.show()

##########################
### Emphasize Edge Way ##
########################

img = cv2.imread('../DATA/sudoku.jpg',0)
#display_img(img)

### SOBEL
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#display_img(sobelx)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
display_2img(sobelx,sobely,'sobel X','sobel Y')

### LAPLACIAN
laplacian = cv2.Laplacian(img,cv2.CV_64F)

### Blended Sobel
blended = cv2.addWeighted(src1=sobelx,alpha=0.5,src2=sobely,beta=0.5,gamma=0)
display_2img(img,blended,'Orgin','Blended Sobel')


### threshold
ret,th1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
display_2img(img,th1,'Orgine','Threshold')


### Gradient
kernel = np.ones((4,4),np.uint8)
gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
display_2img(img,gradient,'Orgine','Gradient')