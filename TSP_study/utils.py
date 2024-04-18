# 存放一些工具函数
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def distance(city1, city2):
    """
    用于计算两个城市之间的欧拉距离
    city1 = [x1, y1], city2 = [x2, y2]
    """
    return np.linalg.norm(np.array(city1) - np.array(city2))

def cls_cities(cities,labels):
    """
    根据不同的标签，对城市进行分类
    cities：城市的坐标
    labels:聚类后返回的标签
    结果如：
    {
        1:[[x1,y1],[x2,y2],..],
        2:[[m1,n1],[m2,n2],..],
        .....
    }
    """
    classified_points=defaultdict(list)
    # 根据标签对点进行分类
    for label,point in zip(labels,cities):
        classified_points[label].append(point)
    # print(classified_points)
    return classified_points


def calcity(c1,c2):
    """
    聚类后，分成几个城市集群
    用于计算两个城市集群之间的最短距离
    输入:
    cities1 = [[x11,y11],[x12,y12],...]
    cities2 = [[x21,y21],[x22,y22],...]
    输出:m,n,min_dist,flag
    m:cities1的连接处
    n:cities2的连接处
    min_dist:两个集群的最小距离
    flag(bool):1代表正序，0代表逆序
    """
    m,n=(-2,-2)  #默认给一个标记
    q=1000000
    temp_dist=0
    min_dist=0
    flag=1
    # 遍历两个城市集群
    for i in range(len(c1)):        # i = 0,1,2 e.g.
        for j in range(len(c2)):    # j = 0,1,2 e.g.
            # 这里可以提出一个算法加快计算速度 ！！！
            # 正序
            temp_dist1 = -distance(c1[i-1],c1[i])-distance(c2[j-1],c2[j])+distance(c1[i-1],c2[j-1])+distance(c1[i],c2[j])
            # 逆序
            temp_dist2 = -distance(c1[i-1],c1[i])-distance(c2[j-1],c2[j])+distance(c1[i],c2[j-1])+distance(c1[i-1],c2[j])
            # 如果正序更小，则选择正序
            if temp_dist1< temp_dist2:
                temp_dist=temp_dist1
            # 逆序
            else:
                temp_dist=temp_dist2
                flag=0
            # 更新最小的连接位置
            if temp_dist<q:
                q=temp_dist
                m=i
                n=j
                min_dist=q
                
    return m,n,min_dist,flag


def cal_total_distance(cities):
    """
    计算闭环回路的总长度
    """
    temp_distance = 0.0
    for i in range(1, len(cities)):
        start, end = cities[i-1], cities[i]     # 从 0 开始 [0,1],..[n-1,n]
        temp_distance += distance(start,end)

    # 回路
    start = cities[0]
    temp_distance += distance(start,end)
    total_distance = temp_distance

    return total_distance


def draw_result(cities):
    """
    cities:绘制结果图像
    """
    best_path_coordinates = cities
    best_path_coordinates.append(best_path_coordinates[0])  # 闭合路径

    x = [coord[0] for coord in best_path_coordinates]
    y = [coord[1] for coord in best_path_coordinates]

    plt.plot(x, y, marker='o', linestyle='-', color='b')
    plt.scatter(x, y, c='red', marker='o', label='Cities')

    # 添加城市标签
    for i, city in enumerate(best_path_coordinates):
        plt.text(city[0], city[1], str(i+1))

    plt.show()

if __name__ == "__main__":
    city1 = [0, 0]
    city2 = [3, 4]
    print(distance(city1,city2))

