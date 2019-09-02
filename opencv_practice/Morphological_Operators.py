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


img = load_img()
#display_img(img)

## Erode the image
kernel = np.ones((5,5),dtype=np.uint8)
erode = cv2.erode(img,kernel,iterations=4)
#display_img(results)
display_2img(img,erode,'orgin','eosion')

## Dilate the image

dilate = cv2.dilate(img,kernel,iterations=4)
display_2img(img,dilate,'orgin','dilate')



img = load_img()
white_noise = np.random.randint(low=0,high=2,size=(600,600))
# Change scale
white_noise = white_noise * 255

noise_img = white_noise + img
#display_img(noise_img)
display_2img(white_noise,noise_img,'white nosie','white noise img')

opening = cv2.morphologyEx(noise_img,cv2.MORPH_OPEN,kernel)
#display_img(opening)
display_2img(img,opening,'orgin','opening')


img = load_img()
black_noise = np.random.randint(low=0,high=2,size=(600,600))
black_noise = black_noise * -255
black_noise_img = img + black_noise
black_noise_img[black_noise_img == -255] = 0
#display_img(black_noise_img)
closing = cv2.morphologyEx(black_noise_img,cv2.MORPH_CLOSE,kernel)
display_2img(black_noise_img,closing,'black noise img','closing')

gradient = cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
display_2img(black_noise_img,gradient,'black noise img','gradient')


