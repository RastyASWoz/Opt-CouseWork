import numpy as np
import time
import random
import matplotlib.pyplot as plt
import json

T_init = 5000.0  # 初始温度
T_end = 1e-3  # 终止温度
alpha = 0.98  # 温度衰减系数
L_max = 1000  # 同一个温度下迭代次数
num = 25  # 城市总数
n =3
city_file = 'city_coordinates.json'

# 生成随机城市坐标并保存到文件
def generate_random_cities(num_of_city):
    city_list = [(random.randint(0, 5000), random.randint(0, 5000)) for _ in range(num_of_city)]
    with open(city_file, 'w') as f:
        json.dump(city_list, f)
    return city_list

# 读取城市坐标
def load_cities():
    try:
        with open(city_file, 'r') as f:
            city_list = json.load(f)
    except FileNotFoundError:
        city_list = generate_random_cities(num)
    return city_list

cityList = load_cities()

def gen_new(planing):
    pos1 = random.randint(0, num - 1)
    pos2 = random.randint(0, num - 1)
    planing[pos2], planing[pos1] = planing[pos1], planing[pos2]
    return planing

def dis(startplaceindex, endplaceindex):
    x1 = cityList[startplaceindex][0]
    y1 = cityList[startplaceindex][1]
    x2 = cityList[endplaceindex][0]
    y2 = cityList[endplaceindex][1]
    return np.sqrt((x1-x2)**2+(y1-y2)**2)

def path_len(planing):
    path = 0 
    for i in range(num-1):
        dis = dis(planing[i],planing[i+1])
        path +=dis
    last_dist = dis(planing[-1],planing[0])
    return path+last_dist

def SA():
    current_plan = list(range(num))
    random.shuffle(current_plan)
    current_distance = path_len(current_plan)
    T = T_init
    distances = []
    iterround = 0

    while T > T_end:
        for _ in range(L_max):
            new_plan = gen_new(current_plan[:])
            new_distance = path_len(new_plan)
            if new_distance < current_distance or random.random() < np.exp((current_distance - new_distance) / T):
                current_plan = new_plan
                current_distance = new_distance
        iterround += 1
        if iterround >= 1000:
            break
        distances.append(current_distance)
        T *= alpha 
        T += n* np.exp(-100 / iterround)

    return current_plan, distances

def plt_history(distances):
    plt.plot(distances)
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.title('Simulated Annealing - Distance vs Iteration')
    plt.show()

if __name__ == "__main__":
    final_plan, distances = SA()
    plt_history(distances)
    print("当前解：", path_len(final_plan))
    final_plan.append(final_plan[0])
    x = [cityList[index][0] for index in final_plan]
    y = [cityList[index][1] for index in final_plan]
    plt.figure(figsize=(15,10))
    plt.plot(x,y,'o')
    plt.plot(x,y,linewidth=1,color='red')
    plt.plot(x[0],y[0],markersize=20)
    plt.show()
    
    # # 将城市坐标plot出来
    # x = [city[0] for city in cityList]
    # y = [city[1] for city in cityList]
    # plt.scatter(x, y)
    # plt.show()
