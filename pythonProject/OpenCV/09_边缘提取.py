import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1 图像读取
img = cv.imread('picture/img.png', 0)

#定义检测水平的核
kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
level = cv.filter2D(img, -1, kernel=kernel)

#定义检测垂直的核
kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
vertical = cv.filter2D(img, -1, kernel=kernel)

sum = cv.imread('picture/img.png', 0)
rows,cols = sum.shape[:2]
for i in range(rows):
    for j in range(cols):
        sum[i][j] = (level[i][j] ** 2 + vertical[i][j] ** 2) ** 0.5

fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(8,8))
axes[0][0].imshow(img,cmap=plt.cm.gray)
axes[0][0].set_title("origin")
axes[0][1].imshow(level,cmap=plt.cm.gray)
axes[0][1].set_title("level")
axes[1][0].imshow(vertical,cmap=plt.cm.gray)
axes[1][0].set_title("vertical")
axes[1][1].imshow(sum,cmap=plt.cm.gray)
axes[1][1].set_title("sum")
plt.show()