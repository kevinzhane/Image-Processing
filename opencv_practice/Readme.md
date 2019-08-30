# Blurring & Smoothing

gamma : 調整圖片的亮度（越高越暗）

np.power(img,gamma=?)
 
cv2.putText(img,text='',org=(),.....): Put the text on Image



Blurring ＆ smoothing : 使用kernel將圖片進行處理，可使圖片變模糊已達到降噪的效果(noise)

step:	build kernel
	use kernel to processing image
	get new image

code1:	kernel = np.ones(shape=(5,5),dtype=np.float32)/25
	dst = cv2.filter2D(img,-1,kernel)

3 different blur on cv2
code2:	cv2.blur(img,ksize=(5,5))
	cv2.GaussianBlur(img,ksize=(),10)
	cv2.medianBlur(img,5)


# Morphological Operators （形態學影像處理）

Morphological Operators: A set of Kernels that can achieve a variety of effects,like reducing noise

Certain operators :	1. Reducing black points (noise) on white background
			2. Achieve an erosion(侵蝕) and dilation（膨脹＆擴大） effect （去除磨損）


Erosion (侵蝕) : The value of the output pixel is the minimum value of all the pixels in the input pixel's neighborhood. In a binary image, if any of the pixels is set to 0, the output pixel is set to 0.

Dilation (擴張): The value of the output pixel is the maximum value of all the pixels in the input pixel's neighborhood. In a binary image, if any of the pixels is set to the value 1, the output pixel is set to 1.

Morphological "opening" : Reducing the noise on the "background"
Morphological "closing" : Reducing the noise on the "foreground"
Morphological "Gradinet" : 調整顏色的方向


# Normal ways to emphasize X & Y axis for doing "Edge Detection"

1. Sobel X & Y
2. Laplacian
3. Threshold
4. Blend Sobel X & Y
5. Gradient
6. Morphology 
7. Combine above ways 








