import numpy as np
external_parameters = np.zeros((4,4),dtype='float64')
internal_parameters = np.zeros((3,4),dtype='float64')
internal_parameters[0][0] = 367.535
internal_parameters[1][1] = 367.535
internal_parameters[0][2] = 260.166
internal_parameters[1][2] = 205.197
external_parameters[3][3] = 1



# R，旋转矩阵
R = np.array([(0.999853, -0.00340388, 0.0167495),
              (0.00300206, 0.999708, 0.0239986),
              (-0.0168257, -0.0239459, 0.999571)])
# print(R.shape)
rows,cols = R.shape[:2]
for i in range(rows):
    for j in range(cols):
        external_parameters[i][j] = R[i][j]
# T 平移矩阵
T = np.array([15.2562, 70.2212, -10.9926]).T
for i in range(0,3):
    external_parameters[i][3] = T[i]


print(external_parameters)
print(internal_parameters)

change = np.dot(internal_parameters,external_parameters)

print(change)

internal_parameters = np.zeros((3,4),dtype='float64')

world = np.array([(0, 0, 0, 1),
              (1, 2, 0, 1),
              (4, 3, 0, 1),
              (6, 7, 0, 1),
              (8, 2, 0, 1),
              (3, 2, 0, 1),
              (4, 1, 0, 1),
              (5, 6, 0, 1)])

rows,cols = world.shape[:2]
for i in range(rows):
    print(np.dot(change, world[i]))