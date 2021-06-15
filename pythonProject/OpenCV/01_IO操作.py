import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread('picture/1.png')
# 2 显示图像
# 2.1 利用opencv展示图像
cv.imshow('image',img)
cv.waitKey(0)
cv.destroyAllWindows()
# 2.2 在matplotplotlib中展示图像
# plt.imshow(img[:,:,::-1])
# plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
# plt.show()
# k = cv.waitKey(0)
# 3 保存图像
# cv.imwrite('messigray.png',img)