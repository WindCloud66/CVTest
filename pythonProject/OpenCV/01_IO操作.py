import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1 读取图像
img = cv.imread('picture/1.png')
# 要读取的图像
# 读取方式的标志
# cv.IMREAD*COLOR：以彩色模式加载图像，任何图像的透明度都将被忽略。这是默认参数。
# cv.IMREAD*GRAYSCALE：以灰度模式加载图像
# cv.IMREAD_UNCHANGED：包括alpha通道的加载图像模式。
# 可以使用1、0或者-1来替代上面三个标志

# 2 显示图像
# 2.1 利用opencv展示图像
cv.imshow('image',img)
cv.waitKey(0)
# 注意：在调用显示图像的API后，要调用cv.waitKey()给图像绘制留下时间，否则窗口会出现无响应情况，并且图像无法显示出来。
cv.destroyAllWindows()
# 2.2 在matplotplotlib中展示图像
# plt.imshow(img[:,:,::-1])
# plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
# plt.show()
# k = cv.waitKey(0)
# 3 保存图像
# cv.imwrite('messigray.png',img)