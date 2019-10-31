import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread('../opencv_practice/DATA/internal_external.png',0)

plt.imshow(img,cmap='gray')
plt.show()

# Find Contours --- Opencv update (Don't need image parameter,Because new function won't change the original image)
contours,hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

print(hierarchy)

# Build the same shape of image
external_contours = np.zeros(img.shape)
internal_contours = np.zeros(img.shape)

# Draw the contours (it depend on what the hierarchy value,the last value is a group of contours)
for i in range(len(contours)):
    
    # External contours
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(external_contours,contours,i,255,-1)
    
    # Internal contours
    else:
        cv2.drawContours(internal_contours,contours,i,255,-1)

plt.imshow(external_contours,cmap='gray')
plt.show()

plt.imshow(internal_contours,cmap='gray')
plt.show()