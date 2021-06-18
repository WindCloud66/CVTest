
import cv2
import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
import functools
import skimage.segmentation
from skimage.color import rgb2yiq

img = cv2.imread("picture/img.png")  # read in black & white

segment_mask1 = skimage.segmentation.felzenszwalb(img, scale=100)
segment_mask2 = skimage.segmentation.felzenszwalb(img, scale=1000)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1)
ax1.set_xlabel("k=100")
ax2.imshow(segment_mask2)
ax2.set_xlabel("k=1000")
fig.suptitle("Felsenszwalb's efficient graph based image segmentation")
plt.tight_layout()
plt.show()


