# Image Processing Basic

## 1.1 Blurring & Smoothing

gamma : 調整圖片的亮度（越高越暗）

np.power(img,gamma=?)
 
cv2.putText(img,text='',org=(),.....): Put the text on Image

Blurring ＆ smoothing : 使用kernel將圖片進行處理，可使圖片變模糊已達到降噪的效果(noise)

step:   
1.build kernel  
2.use kernel to processing image  
3.get new image  

code:  	
kernel = np.ones(shape=(5,5),dtype=np.float32)/25  
dst = cv2.filter2D(img,-1,kernel)  

3 different blur on cv2  
code:  
cv2.blur(img,ksize=(5,5))  
cv2.GaussianBlur(img,ksize=(),10)  
cv2.medianBlur(img,5)  


## 1.2 Thresholding (設定閥值) 

Thresholding can be used to create "binary" images (grayscale ---> binary)
類似分類器的概念，透過設定一個閥值，將 > value 設定成 "1" ,反之則設定成 "0"

Multiband thresholding (for 'RGB' image): Color images can also be thresholded,but it need to change color space to 'HSL' or 'HSV' model for convenience.
					  (There is some other model can be thresholded,check wiki 'https://en.wikipedia.org/wiki/Thresholding_(image_processing)')


## 1.3 Morphological Operators （形態學影像處理）

Morphological Operators: A set of Kernels that can achieve a variety of effects,like reducing noise

Certain operators :	1. Reducing black points (noise) on white background
			2. Achieve an erosion(侵蝕) and dilation（膨脹＆擴大） effect （去除磨損）


Erosion (侵蝕) : The value of the output pixel is the minimum value of all the pixels in the input pixel's neighborhood. In a binary image, if any of the pixels is set to 0, the output pixel is set to 0.

Dilation (擴張): The value of the output pixel is the maximum value of all the pixels in the input pixel's neighborhood. In a binary image, if any of the pixels is set to the value 1, the output pixel is set to 1.

Morphological "opening" : Reducing the noise on the "background"  
Morphological "closing" : Reducing the noise on the "foreground"  
Morphological "Gradient" : 調整顏色的方向  


## Normal ways to emphasize X & Y axis for doing "Edge Detection"

1. Sobel X & Y
2. Laplacian
3. Threshold
4. Blend Sobel X & Y
5. Gradient
6. Morphology 
7. Combine above ways 



## 1.4 Histogram(顯示圖形值的分佈---直方圖)

Histogram Equalization:  
Method of contrast adjustment based on the image histogram (Reduce the color depth)  
(將圖形中的值最小與最大擴展成0&255,使histogram看起來更加線性,增加對比程度) (low constrast---->high constrast)  

1. Build the mask
2. Use mask to perform parts histogram equlization
3. Use "cv2.calHist" to show histogram 

Full color Histogram Equlization
(不能分別對單一頻道執行均衡化，因此須轉成其他色域進行均衡化)
1. Color channel BGR ---> HSV
2. Use "cv2.equlizeHist" to insteads orgin value
3. Color channel HSV ---> RGB

More details:
Histogram equalization is a non-linear process. Channel splitting and equalizing each channel separately is not the proper way for equalization of contrast. Equalization involves Intensity values of the image not the color components. So for a simple RGB color image, HE should not be applied individually on each channel. Rather, it should be applied such that intensity values are equalized without disturbing the color balance of the image. So, the first step is to convert the color space of the image from RGB into one of the color spaces which separate intensity values from color components. 



# Object Detection 

## 1 Template Matching

It simply scans a larger images for a provided template by sliding the template target image across the larger image

Use 'cv2.matchTemplate' & 'cv2.minMaxLoc' to build the Headmap and draw the rectangle for face dectection

six methods:
1. 'cv2.TM_CCOEFF' 
2. 'cv2.TM_CCOEFF_NORMED' 
3. 'cv2.TM_CCORR' 
4. 'cv2.TM_CCORR_NORMED'
5. 'cv2.TM_SQDIFF' 
6. 'cv2.TM_SQDIFF_NORMED

## 2 Corner Detection

### Harris Corner Detection

Swifting the 'window' to check the image if it have large change

Flat regions: Have no change in all directions
Edges: Won't have a major change long the direction of the edge

Use 'cv2.cornerHarris' to find the corner point & use dilate to expand the point

### Shi-Tomasi Detection

1. Use 'cv2.goodFeaturesToTrack' to find the corner on the image
2. Change the 'float' to 'int'
3. Draw the circle point (corner) on the image 


## 3 Edge Detection

### Canny Edge Detection Process

1. Apply Gaussian filter to smooth the image in order to remove the noise ---> Find the intensity gradients of the image
2. Apply non-maximum suppression(NMS,非極大值抑制) to get rid of spurious response(偽訊號響應) to edge detection
(類似過濾掉不需要的資訊，只取我們需要的部份) src:https:'//www.cnblogs.com/makefile/p/nms.html'
3. Apply double threshold to determine potential edges
4. Track edge by hysteresis: Finalize the edtection of edges by suppressing(抑制）all the other edges that are weak and not connected to strong edges

Need to adjust image to decide the 'low' & 'high' value on threshold

Use 'cv2.canny' to find edges,performing blurred first can reduce some noise and find strong edges


## 4 Grid Detection (格子)

Two convenient function
1. 'cv2.findChessboardCorners'
2. 'cv2.findCirclesGrid'


## 5 Contours Detection (輪廓)

Contours: A curve joining all the continuous points,contours are a useful tool for shape analysis & object detection and recognition

1. Use 'cv2.findContours' to get the hierarchy(層次) & contours & image
2. Build the same shape of the original image
3. Draw the contours on the image

Hierarchy: It will group different contours from values
1.External contours: '-', like:-1
2.Internal contours: '+', like:0,4

## 6 Feature Matching

Feature matching defining key features from an input image(Using ideas from corner,edge,and contours detection). Then using a distance calculation,finds all the matches in a secondary image.

3 methods:
1. Brute-Force Matching with ORB descriptors
2. Brute-Force Matching with SIFT descriptors and Ratio Test
3. FLANN based Matcher (fast than the brute force methods,but it just find the approximate nearest neighbors,it mean is a good matching but not best)

Model:
1. Build the create object
2. Find the key point and descriptor
3. Create the matching object and sort the matching value
4. Draw the match


## 7 Watershed Algorithm (分水嶺)

(可以利用圖片的顏色對物體進行分類)
Any grayscale image can be viewed as a topographic surface where 'high intensity' denotees peaks and hills while 'low intensity' denotes valleys.

The algorithm can then fill every isolated vallyes(local minima) with different colored water(label).

Use watershed to distinct the segments
Step:
1. Blur the image(Try strong blur to for the lagre details image)
2. Grayscale and Threshold (2 types of threshold: Binary_INV or OTSU)
3. We face the problem (the threshold image look like all connect to each other)
4-1. 'Remove the noise' or 'erosion'--- normal methods
4-2. Find the sure_background use 'dilation' & sure_foreground use 'Distance Transform' (利用膨脹找到sure_bk)
5. Use watershed to solve 'unknown region'
6. Use connected components to grab some markers
7. Now have 'sure_background' & 'unknow region' and 'six markers points'
8. Apply watershed algorithm ---> Get six distinct point on segments
9. Find the contour and draw it 

## 8 Face Detection (Haar-like feature)

1.Import the the trained .xml file (cv2.CascadeClassifier)
2.Scan the image to find feature rectangle axis(.detectMultiScale)
3.You can adjust the parameter of above function (scaleFactor & minNeighbor)
4.Draw the rectange on the origin image
5.Practice for blur this area(car plate)

# Object Tracking

Two assumption:
1.Pixel intensities of an object don't change between two consecutive frames
2.Neighbouring pixel have similar motion

## Optical flow (光流法)

利用2張不同的連續影像，去實現判斷影像的移動軌跡，在找特徵點時需要注意

1. Lucas-Kanade: 通過金字塔Lucas-Kanade 光流方法計算某些點集的光流（稀疏光流）
2. Gunnar Farneback: 用Gunnar Farneback 的算法計算稠密光流（即圖像上所有像素點的光流都計算出來）
3. Horn-Schunck: 通過塊匹配的方法來計算光流。
4. SimpleFlow: http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/



## MeanShift and CamShift Tracking Theory 

MeanShift: First it will start at initial position,then through the iteration it will more close the clusters(like knn). After subsequent iterations,the cluster means have stopped moving.(You can't contorl the number of cluster,it will calculate by itself)
 
In image: Meanshift can be given a target to track,calculate the color histogram of the target area,and then keep sliding the tracking window to the closet match(the cluster center).......If just use MeanShift on video,it can't track the target when the object is moving.

In Video: We use 'CAMshift' to update the size of the window.



src:https://stackoverflow.com/questions/4831813/image-segmentation-using-mean-shift-explained





