### Blending Image ####
### Use add Weighted function ####
## The blending image often ues simple formula:
## ---->  new_pixel = a * pixel_1 + b * pixel_2 + y


import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

## It's have different shape
print (img1.shape)
#print (img2.shape)

# Blending images of the same size
img1_1 =  cv2.resize(img1,(600,600))
img2_1 =  cv2.resize(img2,(600,600))

# Only applicate on same image size
blended = cv2.addWeighted(src1=img1_1,alpha=0.5, src2=img2_1, beta=0.5,gamma=0) 

plt.subplot(2,2,1)
plt.imshow(img1)
plt.subplot(2,2,2)
plt.imshow(img2)
plt.subplot(2,2,3)
plt.imshow(blended)
plt.show()


#####################################################################

# Overlay small image on top of a larger image (no blending)
# ROI
img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
img2 = cv2.resize(img2,(600,600))

large_img = img1
small_img = img2

x_offset = 0
y_offset = 0

x_end = x_offset + small_img.shape[1]
y_end = y_offset + small_img.shape[0]

large_img[y_offset:y_end,x_offset:x_end] = small_img

plt.imshow(large_img)
plt.show()


########################################################################
# Blend together image of different sizes
#Do the Water Marks
# -----> Img1 + Mask ---> Img2 with masked Img1



img1 = cv2.imread('../DATA/dog_backpack.png')
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)


img2 = cv2.imread('../DATA/watermark_no_copy.png')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img2 = cv2.resize(img2,(600,600))
img2gray = cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

x_offset = 934 - 600
y_offset = 1401 - 600

roi = img1[y_offset:1401,x_offset:934]

# Inverse the dark to white
mask_inv = cv2.bitwise_not(img2gray)

import numpy as np

# Build white chanel
white_background = np.full(img2.shape,255,dtype=np.uint8)

# Incress shape from (600,600,1) to (600,600,3) use MASK
bk = cv2.bitwise_or(white_background,white_background,mask=mask_inv)

fg = cv2.bitwise_or(img2,img2,mask=mask_inv)

final_roi = cv2.bitwise_or(roi,fg)

large_img = img1
small_img = final_roi

# push the Small image(final_roi) on Large image
large_img[y_offset:y_offset+small_img.shape[0],x_offset:x_offset+small_img.shape[1]] = small_img

plt.subplot(2,2,1)
plt.imshow(mask_inv,cmap="gray")
plt.subplot(2,2,2)
plt.imshow(fg)
plt.subplot(2,2,3)
plt.imshow(final_roi)
plt.subplot(2,2,4)
plt.imshow(large_img)
plt.show()

#### Key point ####
# 1. Blending with same size image ----> cv2.addWeighted(src1,alpha ,src2,beta, gamma)
# 2, OverLay (覆蓋): put the Small image on top of Large image
# 3, WaterMaker: Build ROI(類似遮罩，把需要的範圍選取起來進行處理)
#    利用 cv2.bitwise_not 將黑白顏色進行轉置，注意轉換完的small image 只有 1 的維度
#    ，因此需要利用 cv2.bitwise_or 將 轉置完的watermark 放到 3 個全白的 chanel 上
#    ，使其變成rgb 的維度，最後將處理好的圖片覆蓋上原圖，完成浮水印的製作
#