# 开发时间: 2022/4/8 10:48
# _*_coding=utf8_*_

# """
# 欧氏距离
#
# A(13,26)
# B(56,89)
# """
#
# #以numpy的方式实现
# import numpy as np
# from scipy.spatial.distance import pdist
#
# x = np.array([13,56])  #转化为数组
# y = np.array([26,89])  #转化为数组
#
# _x = np.mat(x)    #将数组转化为矩阵
# _y = np.mat(y)    #将数组转化为矩阵
#
# dist_np = np.sqrt(np.sum(np.square(_x - _y)))   #差的平方的和的开方
#
# print("distance for numpy",dist_np)
#
#
#
# #以scipy方式实现
# z = np.vstack([x,y]) #将x,y进行数组堆叠
# dist_scipy = pdist(z)
# print(dist_scipy)

"""
余弦相似度

A(13,26)
B(-56-,89)
"""

#以numpy的方式实现

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
A = np.mat(np.array([13,26]))
B = np.mat(np.array([-56,-89]))

i = float(A * B.T)   #分子

n = np.linalg.norm(A) * np.linalg.norm(B)  #分母

sim = i / n

print("sim for numpy",sim)


#以sklearn的方式实现
z = np.array([[13,26],[56,89]])
_A = z[0]
_B = z[1]
sim_sklearn = cosine_similarity(z)

print("sim for sklearn",sim_sklearn[0][1])