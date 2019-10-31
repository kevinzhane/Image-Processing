import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('../opencv_practice/DATA/sammy_face.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 1. Change the threshold value to get better result
edges = cv2.Canny(image=img,threshold1=127,threshold2=127)

plt.imshow(img)
plt.title('Original')
plt.show()
plt.imshow(edges)
plt.title('Canny Threshold values: 127,127')
plt.show()

edges = cv2.Canny(image=img,threshold1=0,threshold2=255)

plt.imshow(edges)
plt.title('Canny Threshold values: 0,255')
plt.show()


# Better way to find good threshold values
med_val = np.median(img)

# Lower threshold to either 0 or 70% of the median value whichever is greater
lower = int(max(0,0.7*med_val))

# Upper threshold to either 130% of the median or the max 255 value whichever is smaller
upper = int(min(255,1.3*med_val))

edges = cv2.Canny(image=img,threshold1=lower,threshold2=upper+100)

plt.imshow(edges)
plt.title('Canny Median Threshold values:' + str(lower) + ',' + str(upper+100))
plt.show()


# 2.  Bluring the image to get rid of certain details (Best way)

blurred_img = cv2.blur(img,ksize=(5,5))
edges = cv2.Canny(image=blurred_img,threshold1=lower,threshold2=upper+50)

plt.imshow(edges)
plt.title('k=5 Canny Blurred Threshold values:' + str(lower) + ',' + str(upper+50))
plt.show()

blurred_img = cv2.blur(img,ksize=(7,7))
edges = cv2.Canny(image=blurred_img,threshold1=lower,threshold2=upper+50)

plt.imshow(edges)
plt.title('k=7 Canny Blurred Threshold values:' + str(lower) + ',' + str(upper+50))
plt.show()