import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# 1 读取图像
img = cv.imread("picture/letter.png")
# 2 创建核结构
kernel = np.ones((5, 5), np.uint8)

# 3 图像腐蚀和膨胀
erosion = cv.erode(img, kernel) # 腐蚀
dilate = cv.dilate(img, kernel) # 膨胀

# 4 图像展示
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,8),dpi=100)
axes[0].imshow(img)
axes[0].set_title("origin")
axes[1].imshow(erosion)
axes[1].set_title("erosion")
axes[2].imshow(dilate)
axes[2].set_title("dilate")
plt.show()


img1 = cv.imread("picture/letteropen.png")
img2 = cv.imread("picture/letterclose.png")
# 2 创建核结构
kernel = np.ones((10, 10), np.uint8)
# 3 图像的开闭运算
cvOpen = cv.morphologyEx(img1,cv.MORPH_OPEN,kernel) # 开运算
cvClose = cv.morphologyEx(img2,cv.MORPH_CLOSE,kernel)# 闭运算
# 4 图像展示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img1)
axes[0,0].set_title("origin")
axes[0,1].imshow(cvOpen)
axes[0,1].set_title("cvOpen")
axes[1,0].imshow(img2)
axes[1,0].set_title("origin")
axes[1,1].imshow(cvClose)
axes[1,1].set_title("cvClose")
plt.show()

# 1 读取图像
img1 = cv.imread("picture/letteropen.png")
img2 = cv.imread("picture/letterclose.png")
# 2 创建核结构
kernel = np.ones((10, 10), np.uint8)
# 3 图像的礼帽和黑帽运算
cvOpen = cv.morphologyEx(img1,cv.MORPH_TOPHAT,kernel) # 礼帽运算
cvClose = cv.morphologyEx(img2,cv.MORPH_BLACKHAT,kernel)# 黑帽运算
# 4 图像显示
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(10,8))
axes[0,0].imshow(img1)
axes[0,0].set_title("origin")
axes[0,1].imshow(cvOpen)
axes[0,1].set_title("cvOpen")
axes[1,0].imshow(img2)
axes[1,0].set_title("origin")
axes[1,1].imshow(cvClose)
axes[1,1].set_title("cvClose")
plt.show()