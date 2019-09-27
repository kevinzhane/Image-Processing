import cv2
import numpy as np
import matplotlib.pyplot as plt

road = cv2.imread('../opencv_practice/DATA/road_image.jpg')
road_copy = np.copy(road)

# Build the marker & Segments
marker_image = np.zeros(road.shape[:2],dtype=np.int32) # [600,800]
segments = np.zeros(road.shape,dtype=np.uint8) # -- [600,800,3]

plt.imshow(marker_image)
plt.show()

plt.imshow(segments)
plt.show()