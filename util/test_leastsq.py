# 开发时间: 2022/4/7 22:04
# _*_coding=utf8_*_


import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt(
    "D:\\pycharm\\test_rec\\data\\data.csv",
    dtype=float,  # 数据类型float
    usecols=(0, 1),  # 要使用的列
    encoding="utf8",  # 字符串编码格式
    delimiter=","    #分隔符
)

x = data[:, 0]  # 把第一列作为x轴数据
y = data[:, 1]  # 把第二列作为y轴数据

plt.scatter(x, y, color="red")

plt.show()
