import matplotlib.pyplot as plt
import numpy as np

def DataLoader(path=r'Data/att532.txt',show=False,is_np=False):
    """
    path:文件的路径
    show:是否显示坐标以及在图中的分布位置
    is_np:是否转换为np_array的形式
    """

    cities=[]
    with open(path,mode='r') as f:
        data = f.readlines()
    
    for line in data:
        coords = line.strip().split('\t')
        cities.append([int(coords[0]),int(coords[1])])

    if show:
        x=[c[0] for c in cities]
        y=[c[1] for c in cities]
        plt.scatter(x,y)
        plt.show()

    if is_np:
        # 是否转换为np.array的形式
        cities = np.array(cities)
        return cities
    
    print(f"第一个城市的坐标为{cities[0]}")
    return cities
    


if __name__ == "__main__":
    cities=DataLoader(show=True)
