# 开发时间: 2022/4/11 17:14
# _*_coding=utf8_*_

"""
如何使用固定值/平均数/中位数/众数 进行缺失值填充
"""
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
#获取数据
csv_path = pd.read_csv("E:\\fillna.csv")
df = pd.DataFrame(csv_path)

# #固定值填充
# df['rating'] = df['rating'].fillna(0.0)
# print(df)


# #平均值填充
# df['rating'] = df['rating'].fillna(round(df['rating'].mean(),1))
# print(df)

# #中位数填充
# df['rating'].fillna(round(df['rating'].median(),1),inplace=True)
# print(df)


#众数填充
# mode = df['rating'].dropna().mode().values
# df['rating'].fillna(mode[0],inplace=True)
# print(df)


"""
使用sklearn进行 使用固定值/平均数/中位数/众数 进行缺失值填充
"""
# X = df.iloc[:,3:4].values
# imp = SimpleImputer(missing_values=np.nan,strategy='mean')  #missing_values缺失值，strategy要填充的值
# imp = imp.fit(X)
# df['rating'] = np.around(imp.transform(X),decimals=1)
# print(df)


X = df.iloc[:,3:4].values
imp = SimpleImputer(missing_values=np.nan,strategy='mean')  #missing_values缺失值，strategy要填充的值
df['rating'] = np.around(imp.fit_transform(X),decimals=1)
print(df)