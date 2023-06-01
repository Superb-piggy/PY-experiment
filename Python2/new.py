# -*- coding: utf-8 -*-

###############################################################################
#######################           任务实现             ########################
###############################################################################

# 代码 7-1
import numpy as np
import pandas as pd

airline_data = pd.read_csv("任务程序/data/air_data.csv",
                           encoding="gb18030")  # 导入航空数据
print('原始数据的形状为：', airline_data.shape)

# 去除票价为空的记录
exp1 = airline_data["SUM_YR_1"].notnull()

exp2 = airline_data["SUM_YR_2"].notnull()

exp = exp1 & exp2

print('exp的形状是：', exp.shape)
# 选择所有为1的行，：表示选择所有列
airline_notnull = airline_data.loc[exp, :]

print('删除缺失记录后数据的形状为：', airline_notnull.shape)

# 只保留票价非零的，或者平均折扣率不为0且总飞行公里数大于0的记录。
index1 = airline_notnull['SUM_YR_1'] != 0
index2 = airline_notnull['SUM_YR_2'] != 0
index3 = (airline_notnull['SEG_KM_SUM'] > 0) & \
         (airline_notnull['avg_discount'] != 0)
airline = airline_notnull[(index1 | index2) & index3]

print('删除异常记录后数据的形状为：', airline.shape)

# 代码 7-2
# 选取需求特征
airline_selection = airline[["FFP_DATE", "LOAD_TIME",
                             "FLIGHT_COUNT", "LAST_TO_END",
                             "avg_discount", "SEG_KM_SUM"]]
# 构建L特征
L = pd.to_datetime(airline_selection["LOAD_TIME"]) - \
    pd.to_datetime(airline_selection["FFP_DATE"])
# 转换为字符串类型
L = L.astype("str").str.split()
L = L.str[0]
# 转换为月份
L = L.astype("int") / 30

# 合并特征
airline_features = pd.concat([L, airline_selection.iloc[:, 2:]], axis=1)  # 第一个：选择所有的行切片，第二个2：为从2开始 axis为水平方向

print('构建的LRFMC特征前5行为：\n', airline_features.head())
# 列索引（表项）变为字符串类型
airline_features.columns = airline_features.columns.astype(str)
print(airline_features)

# 代码 7-3
from sklearn.preprocessing import StandardScaler

data = StandardScaler().fit_transform(airline_features)

np.savez('任务程序/tmp/airline_scale.npz', data)

print('标准化后LRFMC五个特征为：\n', data[:5, :]) # 选择所有的列，切出前五行

# -*- coding: utf-8 -*-

###############################################################################
#######################           任务实现             #######################
###############################################################################

# 代码 7-4
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans  # 导入kmeans算法

airline_scale = np.load('任务程序/tmp/airline_scale.npz')['arr_0']

k = 5  # 确定聚类中心数
# 构建模型

kmeans_model = KMeans(n_clusters=k, random_state=123, n_init='auto')
# 运行次数，保证不会出现偶然
# 确定不同聚合中心，选择最好的
fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练

print(kmeans_model.cluster_centers_)  # 查看聚类中心

print(kmeans_model.labels_)  # 查看样本的类别标签

# 统计不同类别样本的数目
r1 = pd.Series(kmeans_model.labels_).value_counts()

print('最终每个类别的数目为：\n', r1)

# 画出雷达图
import matplotlib.pyplot as plt

# 聚类中心
cluster_centers = kmeans_model.cluster_centers_

# 创建雷达图,fig为整个大图，ax为子图，polar为极坐标
fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})

# 数据维度（列）数量，每一个列表有五个数值，即为五维
num_dimensions = len(cluster_centers[0])

# 分割一个圆形空间，并设置角度，角度列表不包含右端点
angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

# 绘制雷达图
for i, center in enumerate(cluster_centers):
    values = np.concatenate((center, [center[0]]))  # 闭合图形
    ax.plot(angles, values, label=f'Cluster {i+1}')
    ax.fill(angles, values, alpha=0.25)

# 添加轴标签
ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Length', 'Frequency', 'Recency', 'Channel', 'Monetary Value'])

# 添加标题和图例
ax.set_title('Radar Chart of Clusters')
# 添加图例到子图中
ax.legend()

# 显示雷达图
plt.show()

from sklearn.metrics import calinski_harabasz_score

airline_scale = np.load('任务程序/tmp/airline_scale.npz')['arr_0']
for i in range(2,15):
    #构建并训练模型
    kmeans = KMeans(n_clusters = i,random_state=123,n_init="auto").fit(airline_scale)
    score = calinski_harabasz_score(airline_scale,kmeans.labels_)
    print('数据聚%d类calinski_harabaz指数为：%f'%(i,score))

# from sklearn.metrics import silhouette_score
# import matplotlib.pyplot as plt
#
# airline_scale = np.load('任务程序/tmp/airline_scale.npz')['arr_0']
# silhouettteScore = []
# for i in range(2, 15):
#     #构建并训练模型
#     kmeans = KMeans(n_clusters=i, random_state=123, n_init="auto").fit(airline_scale)
#     score = silhouette_score(airline_scale, kmeans.labels_)
#     silhouettteScore.append(score)
#     print('数据聚%d类Silhouette_score指数为：%f' % (i, score))
# plt.figure(figsize=(10,6))
# plt.plot(range(2,15),silhouettteScore,linewidth=1.5, linestyle="-")
# plt.savefig("评价结果折线图.png")
# plt.show()

# k = 2  # 确定聚类中心数
# # 构建模型
#
# kmeans_model = KMeans(n_clusters=k, random_state=123, n_init='auto')
# # 运行次数，保证不会出现偶然
# # 确定不同聚合中心，选择最好的
# fit_kmeans = kmeans_model.fit(airline_scale)  # 模型训练
#
# print(kmeans_model.cluster_centers_)  # 查看聚类中心
#
# print(kmeans_model.labels_)  # 查看样本的类别标签
#
# # 统计不同类别样本的数目
# r1 = pd.Series(kmeans_model.labels_).value_counts()
#
# print('最终每个类别的数目为：\n', r1)
#
# # 画出雷达图
# import matplotlib.pyplot as plt
#
# # 聚类中心
# cluster_centers = kmeans_model.cluster_centers_
#
# # 创建雷达图,fig为整个大图，ax为子图，polar为极坐标
# fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': 'polar'})
#
# # 数据维度（列）数量，每一个列表有五个数值，即为五维
# num_dimensions = len(cluster_centers[0])
#
# # 分割一个圆形空间，并设置角度，角度列表不包含右端点
# angles = np.linspace(0, 2 * np.pi, num_dimensions, endpoint=False).tolist()
# angles += angles[:1]  # 闭合图形
#
# # 绘制雷达图
# for i, center in enumerate(cluster_centers):
#     values = np.concatenate((center, [center[0]]))  # 闭合图形
#     ax.plot(angles, values, label=f'Cluster {i+1}')
#     ax.fill(angles, values, alpha=0.25)
#
# # 添加轴标签
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(['Length', 'Frequency', 'Recency', 'Channel', 'Monetary Value'])
#
# # 添加标题和图例
# ax.set_title('Radar Chart of Clusters')
# # 添加图例到子图中
# ax.legend()
#
# # 显示雷达图
# plt.show()