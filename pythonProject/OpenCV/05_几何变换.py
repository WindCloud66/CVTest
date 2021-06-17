import cv2 as cv
import numpy as np
# 1. 读取图片
img1 = cv.imread("picture/1.png")
import matplotlib.pyplot as plt
# 2.图像缩放
# 2.1 绝对尺寸
rows,cols = img1.shape[:2]
res = cv.resize(img1,(2*cols,2*rows),interpolation=cv.INTER_CUBIC)
#src : 输入图像  dsize: 绝对尺寸，直接指定调整后图像的大小
#fx,fy: 相对尺寸，将dsize设置为None，然后将fx和fy设置为比例因子即可
#interpolation：插值方法，

# 2.2 相对尺寸
res1 = cv.resize(img1,None,fx=0.5,fy=0.5)

# 3 图像显示
# 3.1 使用opencv显示图像(不推荐)
# cv.imshow("orignal",img1)
# cv.imshow("enlarge",res)
# cv.imshow("shrink）",res1)
# cv.waitKey(0)

# 3.2 使用matplotlib显示图像
fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(10,8),dpi=100)

axes[0].imshow(res[:,:,::-1])
axes[0].set_title("big")
axes[1].imshow(img1[:,:,::-1])
axes[1].set_title("origin")
axes[2].imshow(res1[:,:,::-1])
axes[2].set_title("small")
plt.show()

# 2. 图像平移
rows,cols = img1.shape[:2]
M = M = np.float32([[1,0,10],[0,1,10]])# 平移矩阵 img: 输入图像 M： 2*3移动矩阵
dst = cv.warpAffine(img1,M,(cols,rows))#dsize: 输出图像的大小
#注意：输出图像的大小，它应该是(宽度，高度)的形式。请记住,width=列数，height=行数。

# 3. 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img1[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("shift")
plt.show()

# 1 读取图像
img = cv.imread("picture/1.png")

# 2 图像旋转
rows,cols = img.shape[:2]
# 2.1 生成旋转矩阵
M = cv.getRotationMatrix2D((cols/2,rows/2),90,1)
#cv2.getRotationMatrix2D(center, angle, scale)
#center：旋转中心     angle：旋转角度      scale：缩放比例

# 2.2 进行旋转变换
dst = cv.warpAffine(img,M,(cols,rows))

# 3 图像展示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img1[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("rotation")
plt.show()



img = cv.imread("picture/1.png")

# 2 仿射变换
rows,cols = img.shape[:2]
# 2.1 创建变换矩阵
pts1 = np.float32([[5,5],[20,5],[5,20]])
pts2 = np.float32([[10,10],[20,5],[10,25]])
M = cv.getAffineTransform(pts1,pts2)
# 2.2 完成仿射变换
dst = cv.warpAffine(img,M,(cols,rows))

# 3 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("result")
plt.show()

img = cv.imread("picture/1.png")
# 2 透射变换
rows,cols = img.shape[:2]
# 2.1 创建变换矩阵
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])

T = cv.getPerspectiveTransform(pts1,pts2)
# 2.2 进行变换
dst = cv.warpPerspective(img,T,(cols,rows))

# 3 图像显示
fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
axes[0].imshow(img[:,:,::-1])
axes[0].set_title("origin")
axes[1].imshow(dst[:,:,::-1])
axes[1].set_title("result")
plt.show()

# 1 图像读取
img = cv.imread("picture/1.png")
# 2 进行图像采样
up_img = cv.pyrUp(img)  # 上采样操作
img_1 = cv.pyrDown(img)  # 下采样操作
# 3 图像显示
cv.imshow('enlarge', up_img)
cv.imshow('original', img)
cv.imshow('shrink', img_1)
cv.waitKey(0)
cv.destroyAllWindows()
