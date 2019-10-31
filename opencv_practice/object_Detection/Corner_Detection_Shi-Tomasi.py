import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../opencv_practice/DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_RGB2GRAY)

real_chess = cv2.imread('../opencv_practice/DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)

gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)

# Find good feature corner
corners = cv2.goodFeaturesToTrack(gray_flat_chess,64,0.01,10)

# Change data type 'float' ----> 'int'
corners = np.int0(corners) 

# Draw the circle on corner point
for i in corners:
    x,y = i.ravel()  # 將多維矩陣降為一維
    cv2.circle(flat_chess,(x,y),3,(0,0,255),-1)
plt.imshow(flat_chess)
plt.show()


# For real chess image
corners = cv2.goodFeaturesToTrack(gray_real_chess,100,0.01,10)
corners = np.int0(corners) 

for i in corners:
    x,y = i.ravel()  
    cv2.circle(real_chess,(x,y),3,(0,0,255),-1)
plt.imshow(real_chess)
plt.show()




