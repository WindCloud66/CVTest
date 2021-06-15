import cv2 as cv
print(cv.__version__)
lena =cv.imread("picture/1.png")
cv.imshow("image",lena)
cv.waitKey(0)