import cv2 as cv
import random
#产生一张椒盐噪声的图片
# png = cv.imread('picture/img.png')
# rows,cols = png.shape[:2]
# for i in range(rows):
#     for j in range(cols):
#         rdn = random.random()
#         if rdn < 0.1:
#             png[i][j] = 0
#             continue
#         rdn = random.random()
#         if rdn < 0.1:
#             png[i][j] = 255
# cv.imwrite('picture/img2.png', png)
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
# 1 图像读取
img = cv.imread('picture/img2.png')
# 2 均值滤波
blur3 = cv.blur(img,(3,3))
blur5 = cv.blur(img,(5,5))
blur7 = cv.blur(img,(7,7))
# cv.blur(src, ksize, anchor, borderType)
# src：输入图像 ksize：卷积核的大小 anchor：默认值 (-1,-1) ，表示核中心 borderType：边界类型

# 3 图像显示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img[:,:,::-1])
axes[0,0].set_title("origin")
axes[0,1].imshow(blur3[:,:,::-1])
axes[0,1].set_title("Box Filter3")
axes[1,0].imshow(blur5[:,:,::-1])
axes[1,0].set_title("Box Filter5")
axes[1,1].imshow(blur7[:,:,::-1])
axes[1,1].set_title("Box Filter7")
plt.show()

# 1 图像读取
img = cv.imread('picture/img2.png')
# 2 高斯滤波
blur3 = cv.GaussianBlur(img,(3,3),1)
blur5 = cv.GaussianBlur(img,(5,5),1)
blur7 = cv.GaussianBlur(img,(7,7),1)
# 3 图像显示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img[:,:,::-1])
axes[0,0].set_title("origin")
axes[0,1].imshow(blur3[:,:,::-1])
axes[0,1].set_title("Gaussian Filter3")
axes[1,0].imshow(blur5[:,:,::-1])
axes[1,0].set_title("Gaussian Filter5")
axes[1,1].imshow(blur7[:,:,::-1])
axes[1,1].set_title("Gaussian Filter7")
plt.show()

# 1 图像读取
img = cv.imread('picture/img2.png')
# 2 中值滤波
blur = cv.medianBlur(img,5)
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8))
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(blur[:,:,::-1])
axes[1].set_title("median Filter")
plt.show()

# 1 图像读取(锐化)
img = cv.imread('picture/img.png')
#定义一个核
kernel = np.array([[-1/9, -1/9, -1/9], [-1/9, 2 - -1/9, -1/9], [-1/9, -1/9, -1/9]], np.float32)
sharpen = cv.filter2D(img, -1, kernel=kernel)

fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,8))
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(sharpen[:,:,::-1])
axes[1].set_title("sharpen")
plt.show()