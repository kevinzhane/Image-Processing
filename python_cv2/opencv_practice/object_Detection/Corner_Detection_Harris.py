import cv2
import numpy as np
import matplotlib.pyplot as plt

flat_chess = cv2.imread('../opencv_practice/DATA/flat_chessboard.png')
flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_BGR2RGB)

gray_flat_chess = cv2.cvtColor(flat_chess,cv2.COLOR_RGB2GRAY)

real_chess = cv2.imread('../opencv_practice/DATA/real_chessboard.jpg')
real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2RGB)

gray_real_chess = cv2.cvtColor(real_chess,cv2.COLOR_BGR2GRAY)




# CornerHarris --- flat chess

gray = np.float32(gray_flat_chess)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

# Expanding the corner point
dst = cv2.dilate(dst,None)

# flat_chess value > 1% of the max value ---> set 255
flat_chess[dst>0.1*dst.max()] = [255,0,0] # RGB

plt.imshow(flat_chess)
plt.show()

# CornerHarris --- real chess

gray = np.float32(gray_real_chess)

dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)

dst = cv2.dilate(dst,None)

# flat_chess value > 10% of the max value ---> set 255
real_chess[dst>0.01*dst.max()] = [255,0,0] # RGB

plt.imshow(real_chess)
plt.show()




