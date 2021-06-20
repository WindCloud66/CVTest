import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.io import imread, imshow
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

#1.读取图片
# color = imread('picture/img.jpg')
#2.Pandas库将每个像素存储为单独的数据点
# def image_to_pandas(image):
#     df = pd.DataFrame([image[:,:,0].flatten(),
#                        image[:,:,1].flatten(),
#                        image[:,:,2].flatten()]).T
#     df.columns = ['Red_Channel','Green_Channel','Blue_Channel']
#     return df
# df_color = image_to_pandas(color)
# df_color.head(5)

#使用K-means算法对图像进行聚类
# kmeans = KMeans(n_clusters=  4, random_state = 42).fit(df_color)
# result = kmeans.labels_.reshape(color.shape[0],color.shape[1])
# fig,axes=plt.subplots(nrows=1,ncols=2,figsize=(8,8))
# axes[0].imshow(color,cmap='viridis')
# axes[0].set_title("origin")
# axes[1].imshow(result,cmap='viridis')
# axes[1].set_title("result")
# plt.show()






#1.加载图片
originImg = cv2.imread('picture/img.png')
# 原始图像的形状
originShape = originImg.shape
# 将图像转换为维度数组
flatImg=np.reshape(originImg, [-1, 3])
# 估计estimate_bandwidth
bandwidth = estimate_bandwidth(flatImg, quantile=0.1, n_samples=100)
ms = MeanShift(bandwidth = bandwidth, bin_seeding=True)
# 运行meanshift
ms.fit(flatImg)
labels=ms.labels_
# 其余的颜色
cluster_centers = ms.cluster_centers_



segmentedImg = cluster_centers[np.reshape(labels, originShape[:2])]
cv2.imshow('Image',segmentedImg.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()





