import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl


x = np.array([0,0.33,0.66,1])

def calculateEquationParameters(x):
    # parameter为二维数组，用来存放参数，sizeOfInterval是用来存放区间的个数
    parameter = []
    sizeOfInterval = len(x) - 1;
    i = 1
    # 首先输入方程两边相邻节点处函数值相等的方程为2n-2个方程
    while i < len(x) - 1:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = x[i] * x[i] * x[i]
        data[(i - 1) * 4 + 1] = x[i] * x[i]
        data[(i - 1) * 4 + 2] = x[i]
        data[(i - 1) * 4 + 3] = 1
        data1 = init(sizeOfInterval * 4)
        data1[i * 4] = x[i] * x[i] * x[i]
        data1[i * 4 + 1] = x[i] * x[i]
        data1[i * 4 + 2] = x[i]
        data1[i * 4 + 3] = 1
        temp = data[2:]
        parameter.append(temp)
        temp = data1[2:]
        parameter.append(temp)
        i += 1
    # 输入端点处的函数值。为两个方程, 加上前面的2n - 2个方程，一共2n个方程
    data = init(sizeOfInterval * 4 - 2)
    data[0] = x[0]
    data[1] = 1
    parameter.append(data)
    data = init(sizeOfInterval * 4)
    data[(sizeOfInterval - 1) * 4] = x[-1] * x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 1] = x[-1] * x[-1]
    data[(sizeOfInterval - 1) * 4 + 2] = x[-1]
    data[(sizeOfInterval - 1) * 4 + 3] = 1
    temp = data[2:]
    parameter.append(temp)
    # 端点函数一阶导数值相等为n-1个方程。加上前面的方程为3n-1个方程。
    i = 1
    while i < sizeOfInterval:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 3 * x[i] * x[i]
        data[(i - 1) * 4 + 1] = 2 * x[i]
        data[(i - 1) * 4 + 2] = 1
        data[i * 4] = -3 * x[i] * x[i]
        data[i * 4 + 1] = -2 * x[i]
        data[i * 4 + 2] = -1
        temp = data[2:]
        parameter.append(temp)
        i += 1
    # 端点函数二阶导数值相等为n-1个方程。加上前面的方程为4n-2个方程。且端点处的函数值的二阶导数为零，为两个方程。总共为4n个方程。
    i = 1
    while i < len(x) - 1:
        data = init(sizeOfInterval * 4)
        data[(i - 1) * 4] = 6 * x[i]
        data[(i - 1) * 4 + 1] = 2
        data[i * 4] = -6 * x[i]
        data[i * 4 + 1] = -2
        temp = data[2:]
        parameter.append(temp)
        i += 1
    return parameter

def init(size):
    j = 0;
    data = []
    while j < size:
        data.append(0)
        j += 1
    return data

def solutionOfEquation(parametes, y):
    sizeOfInterval = len(x) - 1;
    result = init(sizeOfInterval * 4 - 2)
    i = 1
    while i < sizeOfInterval:
        result[(i - 1) * 2] = y[i]
        result[(i - 1) * 2 + 1] = y[i]
        i += 1
    result[(sizeOfInterval - 1) * 2] = y[0]
    result[(sizeOfInterval - 1) * 2 + 1] = y[-1]
    a = np.array(calculateEquationParameters(x))
    b = np.array(result)
    return np.linalg.solve(a, b)

def calculate(paremeters, x):
    result =paremeters[0] * x * x * x + paremeters[1] * x * x + paremeters[2] * x +paremeters[3]
    return result


def cal_mid(y):
    C, H, W = y[0].squeeze(0).shape
    img1 = np.zeros([C, H, W])
    img2 = np.zeros([C, H, W])
    img3 = np.zeros([C, H, W])
    for temp in range(0,4):
        y[temp] = y[temp].squeeze(0).to('cpu')
        y[temp] = y[temp].detach().numpy()
    for i in range(0,C):
        for j in range(0,H):
            for k in range(0,W):
                z = []
                z.append((y[0])[i][j][k])
                z.append((y[1])[i][j][k])
                z.append((y[2])[i][j][k])
                z.append((y[3])[i][j][k])
                result = solutionOfEquation(calculateEquationParameters(x), z)
                x1 = calculate([0, 0, result[0], result[1]], (x[0] + x[1]) / 2)
                img1[i][j][k] = x1
                x2 = calculate([result[2], result[3], result[4], result[5]], (x[1] + x[2]) / 2)
                img2[i][j][k] = x2
                x3 = calculate([result[6], result[7], result[8], result[9]], (x[2] + x[3]) / 2)
                img3[i][j][k] = x3

    # result = solutionOfEquation(calculateEquationParameters(x), y)
    # x1 = calculate([0, 0, result[0], result[1]], (x[0] + x[3]) / 2)
    # if x1<0 or x1>1:
    #     x1 = y[0]
    # x2 = calculate([result[2], result[3], result[4], result[5]], (x[1] + x[2]) / 2)
    # if x2<0 or x2>1:
    #     x2 = y[1]
    # x3 = calculate([result[6], result[7], result[8], result[9]], (x[2] + x[3]) / 2)
    # if x3<0 or x3>1:
    #     x3 = y[2]
    # print(x1,x2,x3)
    # return x1 ,x2 ,x3
    return img1 ,img2 ,img3