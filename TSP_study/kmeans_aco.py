# 最后验证的代码

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ACO import Ant,TSP
import itertools
import time
from dataloader import DataLoader
from utils import distance, cls_cities, calcity, cal_total_distance, draw_result

def cls_data(cities,n=5,show=False):
    """
    用于kmeans分类城市点
    n:分成多少个类别
    返回 centroids labels
    centroids:每个类别的中心点
    labels: 每个城市的标签
    """
    kmeans=KMeans(n_clusters=n,random_state=9)
    y_pred=kmeans.fit_predict(cities)
    # 聚类的中心点
    centroids=kmeans.cluster_centers_
    labels=kmeans.labels_

    if show:
        plt.scatter(cities[:,0],cities[:,1],c=y_pred)
        plt.show()

    return centroids,labels

def cls_algorithm(classified_points,show=True):
    """
    对聚类以后的点群，分别进行ACO算法
    algorithm:选择一种算法
    classified_points:分类以后存储在字典中的点
    返回result_list
    指的是每个点群，各自最短城市序列
    """
    result_list=[]
    for label, points in classified_points.items():
        temp_list=[]
        path=TSP(points).search_path()
        # 存储为字典的形式
        result_list.append([points[i] for i in path])
    # 返回的形式类似于[c1,c2,c3,...]
    if show:
        plt.show()
    return result_list


# 读取数据
if __name__ == "__main__":
    # 加载数据
    cities = DataLoader(is_np=True)
    # 聚类分类
    _,labels = cls_data(cities=cities,n=5)
    # print(labels)
    # 分类得到坐标点
    classified_points = cls_cities(cities=cities,labels=labels)

    result_list=cls_algorithm(classified_points=classified_points,show=True)

    tempc1=result_list[0]  #  第一个城市集群
    cs = result_list[1:]   #  剩下的城市集群

    # 对剩下的城市集群进行排列组合
    permutations=list(itertools.permutations(cs))
    min_total_d=200000
    t1 = time.time()
    # 遍历每一种排列组合情况
    for per in permutations:
        c1=tempc1
        for p in per:

            m,n,min_dist,flag=calcity(c1,p)
            if flag==1:
                c1=c1[:m]+p[n:]+p[:n]+c1[m:]

            # 逆序
            else:
                c1=c1[:m]+p[:n][::-1]+p[n:][::-1]+c1[m:]
        
        total_c=cal_total_distance(c1)
        if total_c<min_total_d:
            min_total_d=total_c
            last_c=c1

        
    t2 = time.time()

    print(f'-----花费了：{int(t2-t1)}s-----')

            
    print("最小距离为：",min_total_d)
    draw_result(cities=last_c)

    

"""
如何找到一个类别中的点到另一个类别中点的最短距离
可以用矩阵的形式存储每个类别的点
然后用计算得到一个类别到另一个类别点的距离，保存为矩阵的形式——>筛选出来两个点
然后用蚁群计算

先用蚁群将每个类别的路线都画出来
然后选择相邻的两个点测距 和上面的方法一样
然后连接

一共是五个步骤：
1.聚类
2.类别之间的蚁群
3.找到最短连接处
4.整体连接
5.保存结果

不足之处：
    在计算两个城市集群之间最短距离的连接处时，花费时间较多，计算复杂度大
    在排列计算城市集群之间组合方式时，用的是枚举法，花费时间多，复杂度呈指数增加
    没有使用多线程并行计算
"""


# 95256
# 95931
# 96298
# 95234
# 96580

# 97696
# 96084

# 96644