import cv2

import math  #加载math库
import numpy as np
import gzip
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def read_image_file(filename, N):
    width = 28
    height = 28
    f = gzip.open(filename, 'r')
    f.read(16)  # 跳过开头16 bytes
    buffer = f.read(width * height * N)  # 读取 N 个28*28二进制数据
    data = np.frombuffer(buffer, dtype='uint8')  # 将二进制的数据转换成Integer
    data = data.reshape(N, width, height)  # reshape to Nx28x28x1
    f.close()
    return data


def read_label_file(filename, N):

    f = gzip.open(filename, 'r')
    f.read(8)  # 跳过 8bytes 开头
    buffer = f.read(N)  #读取 N 个28*28二进制数据
    data = np.frombuffer(buffer, dtype='uint8')  # convert binary data to integers : 0 - 255
    f.close()

    return data

X_5 = read_image_file('data/train-images-idx3-ubyte.gz', 5)
y_5 = read_label_file('data/train-labels-idx1-ubyte.gz', 5)

# print(X_5.shape)
# print(y_5)
# print(X_5[0].shape)


X_train = read_image_file('data/train-images-idx3-ubyte.gz', 5000)
y_train = read_label_file('data/train-labels-idx1-ubyte.gz', 5000)
X_test = read_image_file('data/t10k-images-idx3-ubyte.gz', 1000)
y_test = read_label_file('data/t10k-labels-idx1-ubyte.gz', 1000)

#对x_train进行处理
kernel1 = cv2.getGaborKernel((3, 3), 5, 0, 15, 1, 0, cv2.CV_32F)
kernel1 /= math.sqrt((kernel1 * kernel1).sum())
# fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(10,8))
# axes.imshow(kernel1,cmap=plt.cm.gray)
# axes.set_title("origin")
# plt.show()

imgs1 = np.zeros([5000,28,28],np.uint8)
imgs2 = np.zeros([1000,28,28],np.uint8)
for i in range(0, 5000):
    imgs1[i] = cv2.filter2D(X_train[i], -1, kernel=kernel1)
for i in range(0, 1000):
    imgs2[i] = cv2.filter2D(X_test[i], -1, kernel=kernel1)
X_train = imgs1.reshape(5000, 28*28)
X_test = imgs2.reshape(1000, 28*28)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
print(knn.score(X_test, y_test))