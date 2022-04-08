# 开发时间: 2022/4/7 18:41
# _*_coding=utf8_*_
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot


data = np.loadtxt(
    "D:\\pycharm\\test_rec\\data\\data.csv",
    dtype=float,
    skiprows=1,
    usecols=(0,1),
    delimiter=",",
    encoding="utf8"
)

x = data[:,0]
y = data[:,1]


def func(p,x):
    k,b = p
    y = k * x + b

    return y

def cost(p,x,y):
    tatal_cost = 0
    data_num = len(data)
    for i in range(data_num):
        y_yeal = y[i] #真实值
        _x = x[i]   #样品值
        _y = func(p,_x)#样品计算过的值
        tatal_cost += (_y - y_yeal) ** 2
    return  tatal_cost / data_num

initial_p = [0,0]
learning_rate = 0.0001
max_item = 30
def grad(initial_p,learning_rate,max_item):

    cost_list = []
    k,b = initial_p
    for i in range(max_item):
        k,b = step_grad(k,b,learning_rate,x,y)
        _cost = cost([k,b],x,y)
        cost_list.append(_cost)

    return [k,b,cost_list]

def step_grad(current_k,current_b,learning_rate,x,y):
    sum_grad_k = 0
    sum_grad_b = 0
    data_num = len(data)
    for i in range(data_num):
        y_real = y[i]
        _x = x[i]
        _y = func([current_k,current_b],_x)
        #k,b偏导
        sum_grad_k += (_y - y_real) * _x
        sum_grad_b += _y - y_real
    #k,b梯度
    grad_k = 2 / data_num * sum_grad_k
    grad_b = 2 / data_num * sum_grad_b

    update_k = current_k - learning_rate * grad_k
    update_b = current_b - learning_rate * grad_b

    return update_k,update_b

k,b,cost_list = grad(initial_p,learning_rate,max_item)
# plt.scatter(x,y,color = 'red')
# plt.plot(x,func([k,b],x),color = "blue")

print(cost_list)
plt.plot(cost_list)
plt.show()
