import numpy as np
import cv2 as cv

img = cv.imread('picture/1.png')
# 获取某个像素点的值
px = img[100,100]
# 仅获取蓝色通道的强度值
blue = img[100,100,0]
# 修改某个位置的像素值
img[3,3] = [0,0,255]
# 通道拆分
b,g,r = cv.split(img)
# 通道合并
img = cv.merge((b,g,r))
cv.imshow('image',img)
cv.waitKey(0)