from __future__ import division
import numpy as np
import time
import random
import math
import uuid
# This file is revised for more precise and concise expression.
class V2Vchannels:              
    # Simulator of the V2V Channels
    def __init__(self, n_Veh, n_RB, veh_dict):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2
        self.veh_dict = veh_dict
        self.decorrelation_distance = 10
        self.shadow_std = 3
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([],[])
    def update_positions(self, positions):
        self.positions = positions
    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
    def update_shadow(self, delta_distance_list, remove_idex):
        delta_distance = np.zeros((len(delta_distance_list), len(delta_distance_list)))
        for i in range(len(delta_distance)):
            for j in range(len(delta_distance)):
                delta_distance[i][j] = delta_distance_list[i] + delta_distance_list[j]
        if len(delta_distance_list) == 0: 
            self.Shadow = np.random.normal(0,self.shadow_std, size=(self.n_Veh, self.n_Veh))
        else:
            array = np.delete(self.Shadow, remove_idex, axis=1)
            self.Shadow = np.delete(array, remove_idex, axis=0)
            row_diff = len(self.veh_dict) - self.Shadow.shape[0]
            col_diff = len(self.veh_dict) - self.Shadow.shape[1]
            # 如果需要添加行，则在末尾添加服从均值为0，标准差为3的正态分布的值
            if row_diff > 0:
                row_addition = np.random.normal(loc=0, scale=3, size=(row_diff, self.Shadow.shape[1]))
                self.Shadow = np.vstack([self.Shadow, row_addition])
            # 如果需要添加列，则在末尾添加服从均值为0，标准差为3的正态分布的值
            if col_diff > 0:
                col_addition = np.random.normal(loc=0, scale=3, size=(self.Shadow.shape[0], col_diff))
                self.Shadow = np.hstack([self.Shadow, col_addition])
            self.Shadow = np.exp(-1*(delta_distance/self.decorrelation_distance)) * self.Shadow +\
                         np.sqrt(1 - np.exp(-2*(delta_distance/self.decorrelation_distance))) * np.random.normal(0, self.shadow_std, size = (len(self.veh_dict), len(self.veh_dict)))
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(len(self.veh_dict), len(self.veh_dict), self.n_RB) ) + 1j * np.random.normal(size=(len(self.veh_dict), len(self.veh_dict), self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1,d2)+0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10**9)/(3*10**8)     
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20*np.log10(self.fc/5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc/5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc/5)
        def PL_NLos(d_a,d_b):
                n_j = max(2.8 - 0.0024*d_b, 1.84)
                return PL_Los(d_a) + 20 - 12.5*n_j + 10 * n_j * np.log10(d_b) + 3*np.log10(self.fc/5)
        if min(d1,d2) < 7: 
            PL = PL_Los(d)
            self.ifLOS = True
            self.shadow_std = 3
        else:
            PL = min(PL_NLos(d1,d2), PL_NLos(d2,d1))
            self.ifLOS = False
            self.shadow_std = 4                      # if Non line of sight, the std is 4
        return PL

class V2Ichannels: 
    # Simulator of the V2I channels
    def __init__(self, n_Veh, n_RB, veh_dict):
        self.h_bs = 25
        self.h_ms = 1.5
        self.veh_dict = veh_dict
        self.Decorrelation_distance = 50        
        self.BS_position = [750/2, 1299/2]    # Suppose the BS is in the center
        self.shadow_std = 8
        self.n_Veh = n_Veh
        self.n_RB = n_RB
        self.update_shadow([], self.n_Veh)
    def update_positions(self, positions):
        self.positions = positions
        
    def update_pathloss(self):
        self.PathLoss = np.zeros(len(self.positions))
        for i in range(len(self.positions)):
            d1 = abs(self.positions[i][0] - self.BS_position[0])
            d2 = abs(self.positions[i][1] - self.BS_position[1])
            distance = math.hypot(d1,d2) # change from meters to kilometers
            self.PathLoss[i] = 128.1 + 37.6*np.log10(math.sqrt(distance**2 + (self.h_bs-self.h_ms)**2)/1000)
    def update_shadow(self, delta_distance_list, remove_idex):
        if len(delta_distance_list) == 0:  # initialization
            self.Shadow = np.random.normal(0, self.shadow_std, len(self.veh_dict))
        else:
            self.Shadow = np.delete(self.Shadow, remove_idex, axis=0)
            row_diff = len(self.veh_dict) - self.Shadow.shape[0]
            # 如果需要添加行，则在末尾添加服从均值为0，标准差为3的正态分布的值
            if row_diff > 0:
                row_addition = np.random.normal(loc=0, scale=8, size=(row_diff,))
                self.Shadow = np.concatenate([self.Shadow, row_addition], axis = 0)
            # 如果需要添加列，则在末尾添加服从均值为0，标准差为3的正态分布的值
            delta_distance = np.asarray(delta_distance_list)
            # shadow = [c.V2I_shadow for c in self.veh_dict.values()]
            # shadow = np.asarray(shadow)
            self.Shadow = np.exp(-1*(delta_distance/self.Decorrelation_distance))* self.Shadow +\
                          np.sqrt(1-np.exp(-2*(delta_distance/self.Decorrelation_distance)))*np.random.normal(0,self.shadow_std, len(self.veh_dict))
            # shadow = list(shadow)
            # print('len(self.veh_dict)', len(self.veh_dict))
            # for key, obj in self.veh_dict.items():
            #     # 检查索引是否超出列表范围
            #     if len(shadow) > 0:
            #         # 获取当前索引处的新 shadow 值
            #         new_shadow_value = shadow.pop(0)
            #         # 更新对象的 shadow 属性
            #         obj.V2I_shadow = new_shadow_value
            # self.shadow = [c.V2I_shadow for c in self.veh_dict.values()]
            # self.shadow = np.array(self.shadow)
    def update_fast_fading(self):
        h = 1/np.sqrt(2) * (np.random.normal(size=(len(self.veh_dict), self.n_RB)) + 1j* np.random.normal(size=(len(self.veh_dict), self.n_RB)))
        self.FastFading = 20 * np.log10(np.abs(h))

class Vehicle:
    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []
        self.V2I_shadow = np.random.normal(0, 8)

class Environ:
    # Enviroment Simulator: Provide states and rewards to agents. 
    # Evolve to new state based on the actions taken by the vehicles.
    def __init__ (self, down_lane, up_lane, left_lane, right_lane, width, height):
        self.timestep = 0.02
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height
        self.vehicles_dict = {}
        self.vehicles = []
        self.demands = []  
        self.V2V_power_dB = 23 # dBm
        self.V2I_power_dB = 23 # dBm
        self.V2V_power_dB_List = [23, 10, 5]             # the power levels
        #self.V2V_power = 10**(self.V2V_power_dB)
        #self.V2I_power = 10**(self.V2I_power_dB)
        self.sig2_dB = -114
        self.bsAntGain = 8 
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10**(self.sig2_dB/10) 
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.remove_idex = []
        self.n_RB = 20
        self.n_Veh = 20
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB, self.vehicles_dict)  # number of vehicles
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB, self.vehicles_dict)
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.n_step = 0
        self.p_add_veh = 0

    def add_new_vehicles(self, start_position, start_direction, start_velocity):    
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicle_with_dict(self, start_position, start_direction, start_velocity):
        # 生成一个唯一的标识符
        def generate_unique_id():
            return str(uuid.uuid4())
        vehicle_id = generate_unique_id()  # 假设使用前面提到的 generate_unique_id 函数生成唯一标识符
        # 创建新车辆实例
        new_vehicle = Vehicle(start_position, start_direction, start_velocity)
        # 将新车辆添加到字典中
        self.vehicles_dict[vehicle_id] = new_vehicle
        # 返回新车辆的标识符
        return vehicle_id

    def add_new_vehicles_by_number_dict(self, n):
        for i in range(n):
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            self.add_new_vehicle_with_dict(start_position,start_direction,random.randint(10,15))
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            self.add_new_vehicle_with_dict(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicle_with_dict(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicle_with_dict(start_position,start_direction,random.randint(10,15))
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles_dict), len(self.vehicles_dict)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles_dict))
        self.delta_distance = [vehicle.velocity for vehicle in self.vehicles_dict.values()]

    def add_new_vehicles_by_number(self, n):
        for i in range(n):
            ind = np.random.randint(0,len(self.down_lanes))
            start_position = [self.down_lanes[ind], random.randint(0,self.height)]
            start_direction = 'd'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [self.up_lanes[ind], random.randint(0,self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
            start_position = [random.randint(0,self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position,start_direction,random.randint(10,15))
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity for c in self.vehicles])
        #self.renew_channel()
    def renew_positions(self):
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        #for i in range(len(self.position)):
        while(i < len(self.vehicles)):
            #print ('start iteration ', i)
            #print(self.position, len(self.position), self.direction)
            delta_distance = self.vehicles[i].velocity * self.timestep
            change_direction = False
            if self.vehicles[i].direction == 'u':
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    
                    if (self.vehicles[i].position[1] <=self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),self.left_lanes[j] ] 
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <=self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j] ] 
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >=self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - ( self.vehicles[i].position[1]- self.left_lanes[j])), self.left_lanes[j] ] 
                            #print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False :
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >=self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + ( self.vehicles[i].position[1]- self.right_lanes[j])),self.right_lanes[j] ] 
                                #print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                #print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    
                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):   # came to an cross
                        if (random.uniform(0,1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False :
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0,1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance
            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
            # delete
            #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0],self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1],self.vehicles[i].position[1]]
                
            i += 1

    def renew_positions_and_renew_vehicle(self):
        # ========================================================
        # This function update the position of each vehicle
        # ===========================================================
        i = 0
        to_remove = []
        self.remove_idex = []
        # for i in range(len(self.position)):
        for index, (vehicle_id, vehicle) in enumerate(self.vehicles_dict.items()):
            # print ('start iteration ', i)
            # print(self.position, len(self.position), self.direction)
            delta_distance = vehicle.velocity * self.timestep
            change_direction = False
            if vehicle.direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (vehicle.position[1] <= self.left_lanes[j]) and (
                            (vehicle.position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 0.4):
                            new_x = vehicle.position[0] - (delta_distance - (self.left_lanes[j] - vehicle.position[1]))
                            new_y = self.left_lanes[j]
                            vehicle.position = (new_x, new_y)
                            vehicle.direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (vehicle.position[1] <= self.right_lanes[j]) and (
                                (vehicle.position[1] + delta_distance) >= self.right_lanes[j]):
                            if (random.uniform(0, 1) < 0.4):
                                new_x = vehicle.position[0] + (delta_distance + (self.right_lanes[j] - vehicle.position[1]))
                                new_y = self.right_lanes[j]
                                vehicle.position = (new_x, new_y)
                                vehicle.direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    vehicle.position = list(vehicle.position)
                    vehicle.position[1] += delta_distance
                    vehicle.position = tuple(vehicle.position)
            if (vehicle.direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (vehicle.position[1] >= self.left_lanes[j]) and (
                            (vehicle.position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 0.4):
                            new_x = vehicle.position[0] - (
                                        delta_distance - (self.left_lanes[j] - vehicle.position[1]))
                            new_y = self.left_lanes[j]
                            vehicle.position = (new_x, new_y)
                            vehicle.direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (vehicle.position[1] >= self.right_lanes[j]) and (
                                vehicle.position[1] - delta_distance <= self.right_lanes[j]):
                            if (random.uniform(0, 1) < 0.4):
                                new_x = vehicle.position[0] + (
                                            delta_distance + (vehicle.position[1] - self.right_lanes[j]))
                                new_y = self.right_lanes[j]
                                vehicle.position = (new_x, new_y)
                                vehicle.direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    vehicle.position = list(vehicle.position)
                    vehicle.position[1] -= delta_distance
                    vehicle.position = tuple(vehicle.position)
            if (vehicle.direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (vehicle.position[0] <= self.up_lanes[j]) and (
                            (vehicle.position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 0.4):
                            change_direction = True
                            new_x = self.up_lanes[j]
                            new_y = vehicle.position[1] + (
                                        delta_distance - (self.up_lanes[j] - vehicle.position[0]))
                            vehicle.position = (new_x, new_y)
                            vehicle.direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (vehicle.position[0] <= self.down_lanes[j]) and (
                                (vehicle.position[0] + delta_distance) >= self.down_lanes[j]):
                            if (random.uniform(0, 1) < 0.4):
                                change_direction = True
                                new_x = self.down_lanes[j]
                                new_y = vehicle.position[1] - (
                                        delta_distance - (self.down_lanes[j] - vehicle.position[0]))
                                vehicle.position = (new_x, new_y)
                                vehicle.direction = 'd'
                                break
                if change_direction == False:
                    vehicle.position = list(vehicle.position)
                    vehicle.position[0] += delta_distance
                    vehicle.position = tuple(vehicle.position)
            if (vehicle.direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):
                    if (vehicle.position[0] >= self.up_lanes[j]) and (
                            (vehicle.position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (random.uniform(0, 1) < 0.4):
                            change_direction = True
                            new_x = self.up_lanes[j]
                            new_y = vehicle.position[1] + (
                                    delta_distance - (vehicle.position[0]-self.up_lanes[j]))
                            vehicle.position = (new_x, new_y)
                            vehicle.direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (vehicle.position[0] >= self.down_lanes[j]) and (
                                (vehicle.position[0] - delta_distance) <= self.down_lanes[j]):
                            if (random.uniform(0, 1) < 0.4):
                                change_direction = True
                                new_x = self.down_lanes[j]
                                new_y = vehicle.position[1] - (
                                        delta_distance - (vehicle.position[0] - self.down_lanes[j]))
                                vehicle.position = (new_x, new_y)
                                vehicle.direction = 'd'
                                break
                    if change_direction == False:
                        vehicle.position = list(vehicle.position)
                        vehicle.position[0] -= delta_distance
                        vehicle.position = tuple(vehicle.position)
            # if it comes to an exit
            if (vehicle.position[0] < 0) or (vehicle.position[1] < 0) or (
                    vehicle.position[0] > self.width) or (vehicle.position[1] > self.height):
                # delete
                to_remove.append(vehicle_id)
                self.remove_idex.append(index)
            i += 1
        for vehicle_id in to_remove:
            del self.vehicles_dict[vehicle_id]
        if (random.uniform(0, 1) < self.p_add_veh):
            ind = np.random.randint(0, len(self.down_lanes))
            directions = ['u', 'd', 'l', 'r']
            start_direction = random.choice(directions)
            if start_direction == 'd':
                start_position = [self.down_lanes[ind], random.randint(0, self.height)]
                self.add_new_vehicle_with_dict(start_position, start_direction, random.randint(10, 15))
            elif start_direction == 'u':
                start_position = [self.up_lanes[ind], random.randint(0, self.height)]
                self.add_new_vehicle_with_dict(start_position, start_direction, random.randint(10, 15))
            elif start_direction == 'l':
                start_position = [random.randint(0, self.width), self.left_lanes[ind]]
                self.add_new_vehicle_with_dict(start_position, start_direction, random.randint(10, 15))
            elif start_direction == 'r':
                start_position = [random.randint(0, self.width), self.right_lanes[ind]]
                self.add_new_vehicle_with_dict(start_position, start_direction, random.randint(10, 15))
            self.p_add_veh = 0
        else:
            self.p_add_veh += 0.002

    def test_channel(self):
        # ===================================
        #   test the V2I and the V2V channel 
        # ===================================
        self.n_step = 0
        self.vehicles = []
        n_Veh = 20
        self.n_Veh = n_Veh
        self.add_new_vehicles_by_number_dict(int(self.n_Veh/4))
        step = 1000
        time_step = 0.1  # every 0.1s update
        for i in range(step):
            self.renew_positions_and_renew_vehicle()
            positions = [c.position for c in self.vehicles_dict.values()]
            self.update_large_fading(positions, time_step, self.remove_idex)
            self.update_small_fading()
            print("Time step: ", i)
            print(" ============== V2I ===========")
            print("Path Loss: ", self.V2Ichannels.PathLoss)
            print("Shadow:",  self.V2Ichannels.Shadow)
            print("Fast Fading: ",  self.V2Ichannels.FastFading)
            print(" ============== V2V ===========")
            print("Path Loss: ", self.V2Vchannels.PathLoss[0:3])
            print("Shadow:", self.V2Vchannels.Shadow[0:3])
            print("Fast Fading: ", self.V2Vchannels.FastFading[0:3])

    def update_large_fading(self, positions, time_step, remove_idex):
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        # delta_distance = time_step * np.asarray([c.velocity for c in self.vehicles])
        delta_distance = [vehicle.velocity for vehicle in self.vehicles_dict.values()]
        self.V2Ichannels.update_shadow(delta_distance, self.n_Veh)
        self.V2Vchannels.update_shadow(delta_distance, remove_idex)
    def update_small_fading(self):
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        
    def renew_neighbor(self):   
        # ==========================================
        # update the neighbors of each vehicle.
        # ===========================================
        i = 0
        for vehicle_id, vehicle in self.vehicles_dict.items():
            vehicle.neighbors = []
            vehicle.actions = []
            #print('action and neighbors delete', self.vehicles[i].actions, self.vehicles[i].neighbors)
        Distance = np.zeros((len(self.vehicles_dict),len(self.vehicles_dict)))
        z = np.array([[complex(vehicle.position[0], vehicle.position[1]) for vehicle in self.vehicles_dict.values()]])
        Distance = abs(z.T-z)
        for vehicle_id, vehicle in self.vehicles_dict.items():
            sort_idx = np.argsort(Distance[:,i])
            for j in range(3):
                vehicle.neighbors.append(sort_idx[j+1])
            i += 1
            destination = np.random.choice(sort_idx[1:int(len(sort_idx)/5)],3, replace = False)
            vehicle.destinations = destination
    def renew_channel(self):
        # ===========================================================================
        # This function updates all the channels including V2V and V2I channels
        # =============================================================================
        positions = [c.position for c in self.vehicles_dict.values()]
        self.V2Ichannels.update_positions(positions)
        self.V2Vchannels.update_positions(positions)
        self.V2Ichannels.update_pathloss()
        self.V2Vchannels.update_pathloss()
        delta_distance = 0.05 * np.asarray([c.velocity for c in self.vehicles_dict.values()])
        self.V2Ichannels.update_shadow(delta_distance, self.remove_idex)
        self.V2Vchannels.update_shadow(delta_distance, self.remove_idex)
        self.V2V_channels_abs = self.V2Vchannels.PathLoss + self.V2Vchannels.Shadow + 50 * np.identity(
            len(self.vehicles_dict))
        # print('self.V2Ichannels.Shadow', self.V2Ichannels.Shadow.shape)
        self.V2I_channels_abs = self.V2Ichannels.PathLoss + self.V2Ichannels.Shadow

    def renew_channels_fastfading(self):   
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        self.renew_channel()
        self.V2Ichannels.update_fast_fading()
        self.V2Vchannels.update_fast_fading()
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - self.V2Vchannels.FastFading
        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - self.V2Ichannels.FastFading
        #print("V2I channels", self.V2I_channels_with_fastfading)
        
    def Compute_Performance_Reward_fast_fading_with_power(self, actions_power):   # revising based on the fast fading part
        actions = actions_power.copy()[:,:,0]  # the channel_selection_part
        power_selection = actions_power.copy()[:,:,1]
        Rate = np.zeros(len(self.vehicles_dict))
        Interference = np.zeros(self.n_RB)  # V2V signal interference to V2I links
        for i in range(len(self.vehicles_dict)):
            for j in range(len(actions[i,:])):
                if not self.activate_links[i,j]:
                    continue
                #print('power selection,', power_selection[i,j])  
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]]  - self.V2I_channels_with_fastfading[i, actions[i,j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)  # fast fading

        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles_dict), 3))
        V2V_Signal = np.zeros((len(self.vehicles_dict), 3))
        
        # remove the effects of none active links
        #print('shapes', actions.shape, self.activate_links.shape)
        #print(not self.activate_links)
        actions[(np.logical_not(self.activate_links))] = -1
        #print('action are', actions)
        keys = list(self.vehicles_dict.keys())
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                key_select = keys[indexes[j, 0]]
                vehicle = self.vehicles_dict[key_select]
                receiver_j = vehicle.destinations[indexes[j, 1]]
                # compute the V2V signal links
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i < self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links  
                for k in range(j+1, len(indexes)):                  # computer the peer V2V links
                    #receiver_k = self.vehicles[indexes[k][0]].neighbors[indexes[k][1]]
                    key_select = keys[indexes[k][0]]
                    vehicle = self.vehicles_dict[key_select]
                    receiver_k = vehicle.destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)               
       
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.zeros(self.activate_links.shape)
        V2V_Rate[self.activate_links] = np.log2(1 + np.divide(V2V_Signal[self.activate_links], self.V2V_Interference[self.activate_links]))

        #print("V2V Rate", V2V_Rate * self.update_time_test * 1500)
        #print ('V2V_Signal is ', np.log(np.mean(V2V_Signal[self.activate_links])))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))


         # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_test * 1500    # decrease the demand
        self.test_time_count -= self.update_time_test               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_test         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_test      # compute the time interval left for next transmission

        # --- update the demand ---
        
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape ) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        #print("demand is", self.demand)
        #print('mean rate of average V2V link is', np.mean(V2V_Rate[self.activate_links]))
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        self.activate_links[np.add(early_finish, unqulified)] = False 
        #print('number of activate links is', np.sum(self.activate_links)) 
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        #if self.n_step % 1000 == 0 :
        #    self.success_transmission = 0
        #    self.failed_transmission = 0
        failed_percentage = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)
        # print('Percentage of failed', np.sum(new_active), self.failed_transmission, self.failed_transmission + self.success_transmission , failed_percentage)    
        return V2I_Rate, failed_percentage #failed_percentage

        
    def Compute_Performance_Reward_fast_fading_with_power_asyn(self, actions_power):   # revising based on the fast fading part
        # ===================================================
        #  --------- Used for Testing -------
        # ===================================================
        actions = actions_power[:, :, 0]  # the channel_selection_part
        #print('self.activate_links', self.activate_links)
        power_selection = actions_power[:, :, 1]
        Interference = np.zeros(self.n_RB)   # Calculate the interference from V2V to V2I
        for i in range(len(self.vehicles_dict)):
            for j in range(len(actions[i, :])):
                if not self.activate_links[i, j]:
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i, j]] - \
                                                     self.V2I_channels_with_fastfading[i, actions[i,j]] + \
                                                     self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)

        #print('Interference', Interference)
        self.V2I_Interference = Interference + self.sig2
        V2V_Interference = np.zeros((len(self.vehicles_dict), 3))
        V2V_Signal = np.zeros((len(self.vehicles_dict), 3))
        Interfence_times = np.zeros((len(self.vehicles_dict), 3))
        actions[(np.logical_not(self.activate_links))] = -1
        keys = list(self.vehicles_dict.keys())
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                key_select = keys[indexes[j,0]]
                vehicle = self.vehicles_dict[key_select]
                receiver_j = vehicle.destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0], indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                #V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[0] - self.V2V_channels_with_fastfading[indexes[j][0]][receiver_j][i])/10) 
                if i<self.n_Veh:
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2I_power_dB - \
                    self.V2V_channels_with_fastfading[i][receiver_j][i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)  # V2I links interference to V2V links
                for k in range(j+1, len(indexes)):
                    key_select = keys[indexes[k][0]]
                    vehicle = self.vehicles_dict[key_select]
                    receiver_k = vehicle.destinations[indexes[k][1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[k][0]][receiver_j][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j][0]][receiver_k][i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1               

        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))
        V2I_Signals = self.V2I_power_dB-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
        V2I_Rate = np.log2(1 + np.divide(10**(V2I_Signals/10), self.V2I_Interference[0:min(self.n_RB,self.n_Veh)]))
        #print("V2I information", V2I_Signals, self.V2I_Interference, V2I_Rate)
        
        # -- compute the latency constraits --
        self.demand -= V2V_Rate * self.update_time_asyn * 1500    # decrease the demand
        self.test_time_count -= self.update_time_asyn               # compute the time left for estimation
        self.individual_time_limit -= self.update_time_asyn         # compute the time left for individual V2V transmission
        self.individual_time_interval -= self.update_time_asyn     # compute the time interval left for next transmission

        # --- update the demand ---
        new_active = self.individual_time_interval <= 0
        self.activate_links[new_active] = True
        self.individual_time_interval[new_active] = np.random.exponential(0.02, self.individual_time_interval[new_active].shape) + self.V2V_limit
        self.individual_time_limit[new_active] = self.V2V_limit
        self.demand[new_active] = self.demand_amount
        
        # -- update the statistics---
        early_finish = np.multiply(self.demand <= 0, self.activate_links)        
        unqulified = np.multiply(self.individual_time_limit <=0, self.activate_links)
        # print('np.add(early_finish, unqulified)', np.add(early_finish, unqulified))
        self.activate_links[np.add(early_finish, unqulified)] = False
        self.success_transmission += np.sum(early_finish)
        self.failed_transmission += np.sum(unqulified)
        fail_percent = self.failed_transmission/(self.failed_transmission + self.success_transmission + 0.0001)            
        return V2I_Rate, fail_percent

    def Compute_Performance_Reward_Batch(self, actions_power, idx):    # add the power dimension to the action selection
        # ==================================================
        # ------------- Used for Training ----------------
        # ==================================================
        actions = actions_power.copy()[:, :, 0]           #
        power_selection = actions_power.copy()[:,:,1]   #
        V2V_Interference = np.zeros((len(self.vehicles), 3))
        V2V_Signal = np.zeros((len(self.vehicles), 3))
        Interfence_times = np.zeros((len(self.vehicles), 3))    # 3 neighbors
        # print(actions)
        origin_channel_selection = actions[idx[0], idx[1]]
        actions[idx[0], idx[1]] = 100  # something not relavant
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            #print('index',indexes)
            for j in range(len(indexes)):
                #receiver_j = self.vehicles[indexes[j,0]].neighbors[indexes[j,1]]
                receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                V2V_Signal[indexes[j, 0],indexes[j, 1]] = 10**((self.V2V_power_dB_List[power_selection[indexes[j, 0],indexes[j, 1]]] -\
                self.V2V_channels_with_fastfading[indexes[j,0], receiver_j, i]+ 2*self.vehAntGain - self.vehNoiseFigure)/10) 
                V2V_Interference[indexes[j,0],indexes[j,1]] +=  10**((self.V2I_power_dB- self.V2V_channels_with_fastfading[i,receiver_j,i] + \
                2*self.vehAntGain - self.vehNoiseFigure)/10)  # interference from the V2I links
                
                for k in range(j+1, len(indexes)):   #当前信道中其余V2V通信对对当前考虑的V2V的干扰
                    receiver_k = self.vehicles[indexes[k,0]].destinations[indexes[k,1]]
                    V2V_Interference[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[k,0],indexes[k,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[k,0],receiver_j,i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference[indexes[k,0],indexes[k,1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0],indexes[j,1]]] - \
                    self.V2V_channels_with_fastfading[indexes[j,0], receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    Interfence_times[indexes[j,0],indexes[j,1]] += 1
                    Interfence_times[indexes[k,0],indexes[k,1]] += 1#这两行同时存在，那Interfence_times代表的是一对链路作为干扰源和被干扰方的总次数，如果只想要被干扰的次数，应该删去第二行
                    
        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))  # the number of RB times the power level
        Deficit_list = np.zeros((self.n_RB, len(self.V2V_power_dB_List)))
        for i in range(self.n_RB):
            indexes = np.argwhere(actions == i)
            V2V_Signal_temp = V2V_Signal.copy()            
            #receiver_k = self.vehicles[idx[0]].neighbors[idx[1]]
            receiver_k = self.vehicles[idx[0]].destinations[idx[1]]
            for power_idx in range(len(self.V2V_power_dB_List)):
                V2V_Interference_temp = V2V_Interference.copy()
                V2V_Signal_temp[idx[0],idx[1]] = 10**((self.V2V_power_dB_List[power_idx] - \
                self.V2V_channels_with_fastfading[idx[0], self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure )/10)
                V2V_Interference_temp[idx[0],idx[1]] +=  10**((self.V2I_power_dB - \
                self.V2V_channels_with_fastfading[i,self.vehicles[idx[0]].destinations[idx[1]],i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                for j in range(len(indexes)):
                    receiver_j = self.vehicles[indexes[j,0]].destinations[indexes[j,1]]
                    V2V_Interference_temp[idx[0],idx[1]] += 10**((self.V2V_power_dB_List[power_selection[indexes[j,0], indexes[j,1]]] -\
                    self.V2V_channels_with_fastfading[indexes[j,0],receiver_k, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                    V2V_Interference_temp[indexes[j,0],indexes[j,1]] += 10**((self.V2V_power_dB_List[power_idx]-\
                    self.V2V_channels_with_fastfading[idx[0],receiver_j, i] + 2*self.vehAntGain - self.vehNoiseFigure)/10)
                V2V_Rate_cur = np.log2(1 + np.divide(V2V_Signal_temp, V2V_Interference_temp))
                if (origin_channel_selection == i) and (power_selection[idx[0], idx[1]] == power_idx):
                    V2V_Rate = V2V_Rate_cur.copy()
                V2V_Rate_list[i, power_idx] = np.sum(V2V_Rate_cur)
                Deficit_list[i,power_idx] = 0 - 1 * np.sum(np.maximum(np.zeros(V2V_Signal_temp.shape), (self.demand - self.individual_time_limit * V2V_Rate_cur * 1500)))
        Interference = np.zeros(self.n_RB)  
        V2I_Rate_list = np.zeros((self.n_RB,len(self.V2V_power_dB_List)))    # 3 of power level
        for i in range(len(self.vehicles)):
            for j in range(len(actions[i,:])):
                if (i ==idx[0] and j == idx[1]):
                    continue
                Interference[actions[i][j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] - \
                self.V2I_channels_with_fastfading[i, actions[i][j]] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10) 
        V2I_Interference = Interference + self.sig2
        for i in range(self.n_RB):            
            for j in range(len(self.V2V_power_dB_List)):
                V2I_Interference_temp = V2I_Interference.copy()
                V2I_Interference_temp[i] += 10**((self.V2V_power_dB_List[j] - self.V2I_channels_with_fastfading[idx[0], i] + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure)/10)
                V2I_Rate_list[i, j] = np.sum(np.log2(1 + np.divide(10**((self.V2I_power_dB + self.vehAntGain + self.bsAntGain \
                - self.bsNoiseFigure-self.V2I_channels_abs[0:min(self.n_RB,self.n_Veh)])/10), V2I_Interference_temp[0:min(self.n_RB,self.n_Veh)])))
                     
        self.demand -= V2V_Rate * self.update_time_train * 1500
        self.test_time_count -= self.update_time_train
        self.individual_time_limit -= self.update_time_train
        if self.demand[idx[0], idx[1]] <= 0:
            time_left = self.V2V_limit
        else:
            time_left = self.individual_time_limit[idx[0], idx[1]]
        self.individual_time_limit [np.add(self.individual_time_limit <= 0,  self.demand <= 0)] = self.V2V_limit
        self.demand[self.demand <= 0] = self.demand_amount
        if self.test_time_count == 0:
            self.test_time_count = 10
        return V2I_Rate_list, Deficit_list, time_left

    def Compute_Interference(self, actions):
        # ====================================================
        # Compute the Interference to each channel_selection
        # ====================================================
        V2V_Interference = np.zeros((len(self.vehicles_dict), 3, self.n_RB)) + self.sig2
        keys = list(self.vehicles_dict.keys())
        if len(actions.shape) == 3:
            channel_selection = actions.copy()[:,:,0]
            power_selection = actions[:,:,1]
            channel_selection[np.logical_not(self.activate_links)] = -1
            for i in range(self.n_RB):
                for k in range(len(self.vehicles_dict)):
                    key_select = keys[k]
                    vehicle = self.vehicles_dict[key_select]
                    for m in range(len(channel_selection[k,:])):
                        V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_channels_with_fastfading[i][vehicle.destinations[m]][i] + \
                        2 * self.vehAntGain - self.vehNoiseFigure)/10)
            for i in range(len(self.vehicles_dict)):
                for j in range(len(channel_selection[i,:])):
                    for k in range(len(self.vehicles_dict)):
                        for m in range(len(channel_selection[k,:])):
                            if (i==k) or (channel_selection[i,j] >= 0):
                                continue
                            key_select = keys[k]
                            vehicle = self.vehicles_dict[key_select]
                            V2V_Interference[k, m, channel_selection[i,j]] += 10**((self.V2V_power_dB_List[power_selection[i,j]] -\
                            self.V2V_channels_with_fastfading[i][vehicle.destinations[m]][channel_selection[i,j]] + 2*self.vehAntGain - self.vehNoiseFigure)/10)

        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)
                
        
    def renew_demand(self):
        # generate a new demand of a V2V
        self.demand = self.demand_amount*np.ones((self.n_RB,3))
        self.time_limit = 10
    def act_for_training(self, actions, idx):
        # =============================================
        # This function gives rewards for training
        # ===========================================
        rewards_list = np.zeros(self.n_RB)
        action_temp = actions.copy()
        self.activate_links = np.ones((self.n_Veh,3), dtype = 'bool')
        V2I_rewardlist, V2V_rewardlist, time_left = self.Compute_Performance_Reward_Batch(action_temp,idx)
        self.renew_positions()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions) 
        rewards_list = rewards_list.T.reshape([-1])
        V2I_rewardlist = V2I_rewardlist.T.reshape([-1])
        V2V_rewardlist = V2V_rewardlist.T.reshape([-1])
        V2I_reward = (V2I_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                      np.min(V2I_rewardlist))/(np.max(V2I_rewardlist) -np.min(V2I_rewardlist) + 0.000001)
        V2V_reward = (V2V_rewardlist[actions[idx[0],idx[1], 0]+ 20*actions[idx[0],idx[1], 1]] -\
                     np.min(V2V_rewardlist))/(np.max(V2V_rewardlist) -np.min(V2V_rewardlist) + 0.000001)
        lambdda = 0.1
        #print ("Reward", V2I_reward, V2V_reward, time_left)
        t = lambdda * V2I_reward + (1 - lambdda) * V2V_reward
        # t = 3 * time_left * V2I_reward + (1 - 3 * time_left) * V2V_reward
        # print("time left", time_left)
        # print ("Reward", V2I_reward, V2V_reward, time_left)
        #return t
        return t - (self.V2V_limit - time_left)/self.V2V_limit

    def act_asyn(self, actions):
        self.n_step += 1
        reward = self.Compute_Performance_Reward_fast_fading_with_power_asyn(actions)
        self.Compute_Interference(actions)
        if self.n_step % 10 == 0:
            self.renew_positions_and_renew_vehicle()
            self.renew_variable_size()
            self.renew_neighbor()
            self.renew_channels_fastfading()
        return reward
    def act(self, actions):
        # simulate the next state after the action is given
        self.n_step += 1        
        reward = self.Compute_Performance_Reward_fast_fading_with_power(actions)
        self.renew_positions_and_renew_vehicle()
        self.renew_channels_fastfading()
        self.Compute_Interference(actions)
        return reward

    def renew_variable_size(self):
        self.V2V_Interference_all = self.renew_numpy_3(self.V2V_Interference_all, "zeros")
        self.individual_time_limit = self.renew_numpy_2(self.individual_time_limit, "ones1")
        self.individual_time_interval = self.renew_numpy_2(self.individual_time_interval, "exponential")
        self.UnsuccessfulLink = self.renew_numpy_2(self.UnsuccessfulLink, "zeros")
        self.activate_links = self.renew_numpy_2(self.activate_links, "zeros")
        self.activate_links = self.activate_links.astype(bool)
        self.demand = self.renew_numpy_2(self.demand, "ones2")


    def renew_numpy_2(self, m, distribution):
        m = np.delete(m, self.remove_idex, axis=0)
        row_diff = len(self.vehicles_dict) - m.shape[0]
        # 如果需要添加行，则在末尾添加服从均值为0，标准差为3的正态分布的值
        if distribution == "normal":
            if row_diff > 0:
                row_addition = np.random.normal(loc=0, scale=3, size=(row_diff, m.shape[1]))
                m = np.vstack([m, row_addition])
            # if col_diff > 0:
            #     col_addition = np.random.normal(loc=0, scale=3, size=(m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "exponential":
            if row_diff > 0:
                row_addition = np.random.exponential(0.05, size=(row_diff, m.shape[1]))
                m = np.vstack([m, row_addition])
            # if col_diff > 0:
            #     col_addition = np.random.exponential(0.05, size=(m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "zeros":
            if row_diff > 0:
                row_addition = np.zeros((row_diff, m.shape[1]))
                m = np.vstack([m, row_addition])
            # if col_diff > 0:
            #     col_addition = np.zeros((m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "ones1":
            if row_diff > 0:
                row_addition = self.V2V_limit * np.ones((row_diff, m.shape[1]))
                m = np.vstack([m, row_addition])
            # if col_diff > 0:
            #     col_addition = np.ones((m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "ones2":
            if row_diff > 0:
                row_addition = self.demand_amount * np.ones((row_diff, m.shape[1]))
                m = np.vstack([m, row_addition])
        return m

    def renew_numpy_3(self, m, distribution):
        m = np.delete(m, self.remove_idex, axis=0)
        row_diff = len(self.vehicles_dict) - m.shape[0]
        # 如果需要添加行，则在末尾添加服从均值为0，标准差为3的正态分布的值
        if distribution == "normal":
            if row_diff > 0:
                row_addition = np.random.normal(loc=0, scale=3, size=(row_diff, m.shape[1], m.shape[2]))
                m = np.concatenate([m, row_addition], axis=0)
            # if col_diff > 0:
            #     col_addition = np.random.normal(loc=0, scale=3, size=(m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "exponential":
            if row_diff > 0:
                row_addition = np.random.exponential(0.05, size=(row_diff, m.shape[1], m.shape[2]))
                m = np.concatenate([m, row_addition], axis=0)
            # if col_diff > 0:
            #     col_addition = np.random.exponential(0.05, size=(m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "zeros":
            if row_diff > 0:
                row_addition = np.zeros((row_diff, m.shape[1], m.shape[2]))
                m = np.concatenate([m, row_addition], axis=0)
            # if col_diff > 0:
            #     col_addition = np.zeros((m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        elif distribution == "ones":
            if row_diff > 0:
                row_addition = np.ones((row_diff, m.shape[1], m.shape[2]))
                m = np.concatenate([m, row_addition], axis=0)
            # if col_diff > 0:
            #     col_addition = np.ones((m.shape[0], col_diff))
            #     m = np.hstack([m, col_addition])
        return m

    def new_random_game(self, n_Veh = 0):
        # make a new game
        self.n_step = 0
        self.vehicles = []
        self.vehicles_dict = {}
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh/4))
        self.V2Vchannels = V2Vchannels(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2Ichannels = V2Ichannels(self.n_Veh, self.n_RB, self.vehicles_dict)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.V2V_Interference_all = np.zeros((self.n_Veh, 3, self.n_RB)) + self.sig2
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh, 3))
        self.test_time_count = 10
        self.V2V_limit = 0.1  # 100 ms V2V toleratable latency
        self.individual_time_limit = self.V2V_limit * np.ones((self.n_Veh,3))
        self.individual_time_interval = np.random.exponential(0.05, (self.n_Veh, 3))
        self.UnsuccessfulLink = np.zeros((self.n_Veh,3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_train = 0.01  # 10ms update time for the training
        self.update_time_test = 0.02 # 2ms update time for testing
        self.update_time_asyn = 0.002 # 0.2 ms update one subset of the vehicles; for each vehicle, the update time is 2 ms
        self.activate_links = np.zeros((self.n_Veh, 3), dtype='bool')

    def create_dynamic_env(self, n_Veh = 0):
        self.n_step = 0
        self.remove_idex = []
        self.vehicles = []
        self.vehicles_dict = {}
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number_dict(int(self.n_Veh / 4))
        self.V2Vchannels = V2Vchannels(len(self.vehicles_dict), self.n_RB, self.vehicles_dict)  # number of vehicles
        self.V2Ichannels = V2Ichannels(len(self.vehicles_dict), self.n_RB, self.vehicles_dict)
        self.renew_channels_fastfading()
        self.renew_neighbor()
        self.V2V_Interference_all = np.zeros((len(self.vehicles_dict), 3, self.n_RB)) + self.sig2
        self.demand_amount = 30
        self.demand = self.demand_amount * np.ones((self.n_Veh, 3))
        self.test_time_count = 10
        self.V2V_limit = 0.1  # 100 ms V2V toleratable latency
        self.individual_time_limit = self.V2V_limit * np.ones((len(self.vehicles_dict), 3))
        self.individual_time_interval = np.random.exponential(0.05, (len(self.vehicles_dict), 3))
        self.UnsuccessfulLink = np.zeros((len(self.vehicles_dict), 3))
        self.success_transmission = 0
        self.failed_transmission = 0
        self.update_time_test = 0.05
        self.update_time_asyn = 0.005
        self.activate_links = np.zeros((len(self.vehicles_dict), 3), dtype='bool')


if __name__ == "__main__":
    up_lanes = [3.5/2,3.5/2 + 3.5,250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]
    down_lanes = [250-3.5-3.5/2,250-3.5/2,500-3.5-3.5/2,500-3.5/2,750-3.5-3.5/2,750-3.5/2]
    left_lanes = [3.5/2,3.5/2 + 3.5,433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]
    right_lanes = [433-3.5-3.5/2,433-3.5/2,866-3.5-3.5/2,866-3.5/2,1299-3.5-3.5/2,1299-3.5/2]
    width = 750
    height = 1299
    Env = Environ(down_lanes,up_lanes,left_lanes,right_lanes, width, height) 
    Env.test_channel()    
