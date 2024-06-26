# -*- coding: utf-8 -*-
import random
import copy
import sys
import matplotlib.pyplot as plt
import numpy as np
import tsplib95
 



# plt.show()
 
# results=[] 

#----------- 蚂蚁 -----------
class Ant(object):
    """
    需要传入ALPHA和BETA
    ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
        ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
    BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
        加快，但是随机性不高，容易得到局部的相对最优
    """
    # 初始化
    def __init__(self,ID,city_num,pheromone_graph,distance_graph,ALPHA=1.0,BETA=2.0):
        
        
        self.ID = ID                 # ID
        self.ALPHA=ALPHA
        self.BETA=BETA
        self.city_num=city_num
        self.distance_graph=distance_graph
        self.pheromone_graph=pheromone_graph
        self.__clean_data()          # 随机初始化出生点
 
    # 初始数据
    def __clean_data(self):
    
        self.path = []               # 当前蚂蚁的路径           
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = [True for i in range(self.city_num)] # 探索城市的状态
        
        # 在这里固定一个起始点
        city_index = random.randint(0,self.city_num-1) # 随机初始出生点
        self.current_city = city_index
        self.path.append(city_index)
        self.open_table_city[city_index] = False
        self.move_count = 1
    
    # 选择下一个城市
    def __choice_next_city(self):
        
        next_city = -1
        select_citys_prob = [0.0 for i in range(self.city_num)]  #存储去下个城市的概率
        total_prob = 0.0
 
        # 获取去下一个城市的概率
        for i in range(self.city_num):
            if self.open_table_city[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    select_citys_prob[i] = pow(self.pheromone_graph[self.current_city][i], self.ALPHA) * pow((1.0/self.distance_graph[self.current_city][i]), self.BETA)
                    total_prob += select_citys_prob[i]
                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))
                    sys.exit(1)
        
        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for i in range(self.city_num):
                if self.open_table_city[i]:
                    # 轮次相减
                    temp_prob -= select_citys_prob[i]
                    if temp_prob < 0.0:
                        next_city = i
                        break
 
        # 未从概率产生，顺序选择一个未访问城市
        # if next_city == -1:
        #     for i in range(city_num):
        #         if self.open_table_city[i]:
        #             next_city = i
        #             break
 
        if (next_city == -1):
            next_city = random.randint(0, self.city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, self.city_num - 1)
 
        # 返回下一个城市序号
        return next_city
    
    # 计算路径总距离
    def __cal_total_distance(self):
        
        temp_distance = 0.0
 
        for i in range(1, self.city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += self.distance_graph[start][end]
 
        # 回路
        end = self.path[0]
        temp_distance += self.distance_graph[start][end]
        self.total_distance = temp_distance
        
    
    # 移动操作
    def __move(self, next_city):
        
        self.path.append(next_city)
        self.open_table_city[next_city] = False
        self.total_distance += self.distance_graph[self.current_city][next_city]
        self.current_city = next_city
        self.move_count += 1
        
    # 搜索路径
    def search_path(self):
 
        # 初始化数据
        self.__clean_data()
 
        # 搜素路径，遍历完所有城市为止
        while self.move_count < self.city_num:
            # 移动到下一个城市
            next_city =  self.__choice_next_city()
            self.__move(next_city)
 
        # 计算路径总长度
        self.__cal_total_distance()
 
#----------- TSP问题 -----------
        
class TSP(object):
    def __init__(self, cities,RHO=0.5,Q=100,ant_num=50):
        self.cities=cities
        distance_x=[city[0] for city in cities]
        distance_y=[city[1] for city in cities]
        self.city_num=len(self.cities)
        self.ant_num=ant_num
        self.RHO=RHO
        self.Q=Q
        #城市距离和信息素
        self.distance_graph = [ [0.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        self.pheromone_graph = [ [1.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        
        # 城市数目初始化为city_num
        self.n = self.city_num
        self.new()
        # 计算城市之间的距离
        for i in range(self.city_num):
            for j in range(self.city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                self.distance_graph[i][j] =float(int(temp_distance + 0.5))

    # 初始化
    def new(self, evt = None):
        # 初始城市之间的距离和信息素
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone_graph[i][j] = 1.0
                
        self.ants = [Ant(ID,city_num=self.city_num,pheromone_graph=self.pheromone_graph,distance_graph=self.distance_graph) for ID in range(self.ant_num)]  # 初始蚁群
        self.best_ant = Ant(-1,self.city_num,pheromone_graph=self.pheromone_graph,distance_graph=self.distance_graph)                          # 初始最优解
        self.best_ant.total_distance = 1 << 31           # 初始最大距离
        self.iter = 1                                    # 初始化迭代次数 
            
    # 开始搜索
    def search_path(self, evt = None):
        self.__running = True 
        while self.__running:
            # 遍历每一只蚂蚁
            for ant in self.ants:
                # 搜索一条路径
                ant.search_path()
                # 与当前最优蚂蚁比较
                if ant.total_distance < self.best_ant.total_distance:
                    # 更新最优解
                    self.best_ant = copy.deepcopy(ant)
            # 更新信息素
            self.__update_pheromone_gragh()
            print (u"迭代次数：",self.iter,u"最佳路径总距离：",int(self.best_ant.total_distance))
            # 连线
            
            
            self.iter += 1

            if self.iter>100:
                break
        
        # results.append(self.best_ant.total_distance)

        self.line(self.best_ant.path)
        # print(self.best_ant.path)
        return self.best_ant.path
 
    def line(self,best_path):
    # 可视化结果
        best_path_coordinates = [self.cities[i] for i in best_path]
        best_path_coordinates.append(best_path_coordinates[0])  # 闭合路径

        x = [coord[0] for coord in best_path_coordinates]
        y = [coord[1] for coord in best_path_coordinates]

        plt.plot(x, y, marker='o', linestyle='-', color='b')
        plt.scatter(x, y, c='red', marker='o', label='Cities')

        # 添加城市标签
        for i, city in enumerate(best_path_coordinates):
            plt.text(city[0], city[1], str(i+1))

        # # 设置图表标题和坐标轴标签
        # plt.title('Ant Colony Optimization for TSP')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')

        # # 显示图表
        # plt.show()
    
    # 更新信息素
    def __update_pheromone_gragh(self):
 
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(self.city_num)] for raw in range(self.city_num)]
        for ant in self.ants:
            for i in range(1,self.city_num):
                start, end = ant.path[i-1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += self.Q / ant.total_distance
                temp_pheromone[end][start] = temp_pheromone[start][end]
 
        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(self.city_num):
            for j in range(self.city_num):
                self.pheromone_graph[i][j] = self.pheromone_graph[i][j] * self.RHO + temp_pheromone[i][j]
 
 
#----------- 程序的入口处 -----------
                
if __name__ == '__main__':

    # 文件路径
    dataPath=r'./data/gr120.tsp'
    # 读取文件，并转换为字典的格式
    data = tsplib95.load(dataPath).as_name_dict()

    # --------------绘制散点图---------------- #
    coordinates=data['display_data']
    # 生成一个二维列表
    coordinates = list({k: v for k, v in sorted(coordinates.items())}.values())

    cities=np.array(coordinates)

    x1 = [city[0] for city in cities]
    y1 = [city[1] for city in cities]

    # plt.plot(x1, y1, marker='o', linestyle='-', color='b')
    plt.scatter(x1, y1, c='green', marker='o', label='Cities')
    results=[]

    # for _ in range(20):
    TSP(cities=cities).search_path()
    plt.show()
    print(results)
    