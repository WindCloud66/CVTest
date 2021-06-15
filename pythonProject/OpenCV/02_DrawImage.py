import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置
# 1 创建一个空白的图像
img = np.zeros((512,512,3), np.uint8)
# 2 绘制图形
# img:要绘制直线的图像    Start,end: 直线的起点和终点
# color: 线条的颜色    Thickness: 线条宽度
cv.line(img,(0,0),(511,511),(255,0,0),5)
# img:要绘制矩形的图像    Leftupper, rightdown: 矩形的左上角和右下角坐标
# color: 线条的颜色    Thickness: 线条宽度
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
# img:要绘制圆形的图像    Centerpoint, r: 圆心和半径   color: 线条的颜色
# Thickness: 线条宽度，为-1时生成闭合图案并填充颜色
cv.circle(img,(447,63), 63, (0,0,255), -1)

font = cv.FONT_HERSHEY_SIMPLEX
# img: 图像 text：要写入的文本数据   station：文本的放置位置
# font：字体     Fontsize :字体大小
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

# 3 图像展示
plt.imshow(img[:,:,::-1])
plt.title('匹配结果'), plt.xticks([]), plt.yticks([])
plt.show()