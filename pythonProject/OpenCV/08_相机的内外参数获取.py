import cv2
import numpy as np
import glob

# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
w = 7
h = 7
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('test/*.jpg')

print('...loading')
for fname in images:
    print(f'processing img:{fname}')
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    print(ret)
    if ret == True:
        print(f'chessboard detected:{fname}')
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

        # 将角点在图像上显示
        # img = cv2.drawChessboardCorners(img, (w,h), corners,ret)
        # img = cv2.drawChessboardCorners(img, (w, h), corners, ret)
        # cv2.namedWindow('img', 0)
        # cv2.resizeWindow('img', 500, 500)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()



ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print (("ret:"),ret)
print (("mtx:\n"),mtx)        # 内参数矩阵
print (mtx.shape)        # 内参数矩阵
print (("rvecs:\n"),rvecs[0])    # 旋转向量  # 外参数
print (len(rvecs))
print (("tvecs:\n"),tvecs[0])    # 平移向量  # 外参数
print (len(tvecs))



