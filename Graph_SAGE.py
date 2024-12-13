import networkx as nx
from Environment import *
import tensorflow as tf
from model_Graph import GraphModel
import re
import matplotlib.pyplot as plt
import os
import tensorflow.keras as keras
class GraphSAGE_sup(keras.Model):
    def __init__(self, environment):
        super().__init__()
        self.env = environment
        self.weight_dir = 'weight'
        self.G_model = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.G_model_target = GraphModel(sample_num=5, depth=2, dims=20, gcn=True, concat=True)
        self.dims = self.G_model.dims
        self.num_vehicle = len(self.env.vehicles)
        self.s_neighs = []
        self.s1_neighs = []
        self.s2_neighs = []
        self.loss = []
        self.saver = None
        self.features = np.zeros((3*self.num_vehicle, 60))
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 1000000
        self.compile_model()
        self.num_V2V_list = np.zeros((len(self.env.vehicles), len(self.env.vehicles)))
        self.link = np.zeros((3*self.num_vehicle, 2))

    def update_target_network(self):
        self.G_model_target.set_weights(self.G_model.get_weights())

    def compile_model(self):
        # 设置学习率衰减
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=self.learning_rate_decay_step,
            decay_rate=self.learning_rate_decay,
            staircase=True
        )

        # 使用闭包来确保能够在优化器中使用动态学习率
        def minimum_lr_fn():
            step_lr = lr_schedule(self.G_model.optimizer.iterations)
            return tf.maximum(step_lr, self.learning_rate_minimum)

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=minimum_lr_fn, rho=0.95, epsilon=0.01)
        self.G_model.compile(optimizer=optimizer, loss='mean_squared_error')

    def build_graph(self, num_V2V_list):
        G = nx.Graph()  # 无向图
        added_nodes_order = []
        #print('vehicles', self.env.vehicles)
        n = np.max(self.env.Distance)
        # 添加节点
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                if num_V2V_list[i, self.env.vehicles[i].destinations[j]] == 1:
                    label = f"{i} to {self.env.vehicles[i].destinations[j]}"
                    self.link[3*i+j, 0] = i
                    self.link[3*i+j, 1] = self.env.vehicles[i].destinations[j]
                    G.add_node(label, features=self.features[3*i+j, :])  # 使用label作为节点的键
                    # 记录添加顺序
                    added_nodes_order.append(label)
        self.link = self.link.astype(int)
        nodes_idx = []
        for m, node_label in enumerate(added_nodes_order):
            nodes_idx.append(m)
        #ordered_node_data = [G.nodes[node_label] for node_label in added_nodes_order]


        # 构建图的结构
        for i in range(len(self.env.vehicles)):
            for j in range(3):
                node_label = f"{i} to {self.env.vehicles[i].destinations[j]}"
                pattern1 = re.compile(rf"\b{i}\b")
                pattern2 = re.compile(rf"\b{self.env.vehicles[i].destinations[j]}\b")
                #if node_label in G:  # 检查节点是否存在于图中
                    #filtered_labels1 = [label for label in G.nodes if str(i) in label]
                    #filtered_labels2 = [label for label in G.nodes if
                                        #str(self.env.vehicles[i].destinations[j]) in label]
                if node_label in G:  # 检查节点是否存在于图中
                    filtered_labels1 = [label for label in G.nodes if pattern1.search(label)]
                    filtered_labels2 = [label for label in G.nodes if pattern2.search(label)]
                    for label in filtered_labels1:
                        if label != node_label:  # 排除自环
                            m_str, _ = label.split(" to ", 1)
                            m = int(m_str)
                            if m == i:
                                edge_weight = 1.0
                            else:
                                Distance1 = self.env.Distance[m, i]
                                edge_weight = (n - Distance1) / n
                            G.add_edge(node_label, label, weight=edge_weight)  # 添加边
                    for label in filtered_labels2:
                        if label != node_label:  # 排除自环
                            m_str, _ = label.split(" to ", 1)
                            m = int(m_str)
                            if m == i:
                                edge_weight = 1.0
                            else:
                                Distance2 = self.env.Distance[m, i]
                                edge_weight = (n - Distance2) / n
                            G.add_edge(node_label, label, weight=edge_weight)  # 添加边
        return G, added_nodes_order, nodes_idx





    def sample(self, all_nodes, nodes, idx):
        s_neighs = []
        s_neighs_idx = []
        s_weights = []
        # print('idx', idx)
        for node_index in idx:
            #print('nodes', nodes)
            node_label = all_nodes[node_index]  # 根据节点索引获取节点标签
            neighs = list(nx.neighbors(self.graph, node_label))  # 根据节点标签获取邻居
            weights = []

            for neighbor in neighs:
                edge_data = self.graph.get_edge_data(node_label, neighbor)
                weight = edge_data['weight'] if 'weight' in edge_data else None
                weights.append(weight)

            #print('neighs', neighs)
            # 采样逻辑
            if self.G_model.sample_num > len(neighs):
                # sampled_neighs = list(np.random.choice(neighs, self.G_model.sample_num, replace=True))
                sampled_indices = np.random.choice(len(neighs), self.G_model.sample_num, replace=True)
                # 根据采样索引获取采样的邻居和权重
                sampled_neighs = [neighs[i] for i in sampled_indices]
                sampled_weights = [weights[i] for i in sampled_indices]

            else:
                # sampled_neighs = list(np.random.choice(neighs, self.G_model.sample_num, replace=False))
                sampled_indices = np.random.choice(len(neighs), self.G_model.sample_num, replace=False)
                # 根据采样索引获取采样的邻居和权重
                sampled_neighs = [neighs[i] for i in sampled_indices]
                sampled_weights = [weights[i] for i in sampled_indices]
            if self.G_model.gcn:
                sampled_neighs.append(node_label)
                sampled_weights.append(1.0)
            s_neighs.append(sampled_neighs)
            s_weights.append(sampled_weights)
            s_neighs_idx.append([all_nodes.index(neigh) for neigh in sampled_neighs])
        return s_neighs, s_neighs_idx, s_weights

    def fetch_batch(self, nodes, idx):
        #print('nodes', nodes)
        self.s1_neighs = []
        self.s2_neighs = []
        self.s2_neighs_idx = []
        self.s2_weights = []
        s1_neighs, s1_neighs_idx, s1_weights= self.sample(nodes, nodes, idx)
        #print('s1_neighs', s1_neighs_idx)
        #print('idx2', len(s1_neighs_idx))
        for neigh in s1_neighs_idx:
            s2_neighs, s2_neighs_idx, s2_weights= self.sample(nodes, s1_neighs, neigh)
            self.s2_neighs.append(s2_neighs)
            self.s2_neighs_idx.append(s2_neighs_idx)
            self.s2_weights.append(s2_weights)

        return s1_neighs_idx, self.s2_neighs_idx, s1_weights, self.s2_weights


    def load_graph(self, graph, order_nodes):
        self.graph = graph
        self.order_nodes = order_nodes

    def test_Complete(self, channel_reward, step, idx, train_flag = True):
        neighs_1_weights = []
        neighs_2_weights = []
        channel_reward = tf.convert_to_tensor(channel_reward, dtype=tf.float32)
        first_order_neighs = np.arange(3*self.num_vehicle)
        second_order_neighs = np.arange(3*self.num_vehicle)
        second_order_neighs = np.tile(second_order_neighs, (3*self.num_vehicle, 1))
        first_order_neighs = first_order_neighs.reshape(1, -1)  # -1告诉NumPy自动计算这个维度的大小
        second_order_neighs = second_order_neighs.reshape(1,60,-1)
        # print('idx', idx)
        inputs = (self.features, idx, first_order_neighs, second_order_neighs)
        agg_result = self.G_model.call_without_weights(inputs)
        agg_result = agg_result.numpy()
        print(agg_result)
        return agg_result

    def get_all_weights(self, node_indices, added_nodes_order):
        all_first_layer_weights = []
        all_second_layer_weights = []

        for node_index in node_indices:
            node_label = added_nodes_order[node_index]
            n = len(added_nodes_order)
            first_layer_weights = []
            second_layer_weights = []

        return all_first_layer_weights, all_second_layer_weights

    def test_get_neighs(self,idx):
        first_order_neighs, second_order_neighs, s1_weights, s2_weights = self.fetch_batch(self.order_nodes, idx)
        return first_order_neighs, second_order_neighs
    def test_unComplete(self, idx, first_order_neighs, second_order_neighs):
        inputs = (self.features, idx, first_order_neighs, second_order_neighs)
        agg_result = self.G_model.call_without_weights(inputs)
        agg_result = agg_result.numpy()
        return agg_result

    def get_edge_weights(self, idx, neighs):
        extracted_data = self.distance[idx, neighs]

    def use_GraphSAGE(self, channel_reward, step, idx, train_flag = True):
        channel_reward = tf.convert_to_tensor(channel_reward, dtype=tf.float32)
        first_order_neighs, second_order_neighs, s1_weights, s2_weights = self.fetch_batch(self.order_nodes, idx)
        # print("first_order_neighs:", first_order_neighs)
        # print("second_order_neighs:", second_order_neighs)
        # print("s1_weights:", s1_weights)
        # print("s2_weights:", s2_weights)
        inputs = (self.features, idx, first_order_neighs, second_order_neighs, s1_weights, s2_weights)
        if train_flag == True and step % 50 == 0 and step > 0 :
            with tf.GradientTape() as G_tape:
                agg_result = self.G_model.call(inputs)
                agg_result_target = self.G_model_target(inputs)
                # agg_result_target = 0.3*agg_result_target + channel_reward
                # tf.print('agg_result first row:', agg_result[0])
                # # print('agg_result_target', agg_result_target)
                # distance_tensor = tf.norm(agg_result - agg_result_target, axis=1)
                difference = agg_result - 0.5 * agg_result_target - 0.5 * channel_reward
                # 计算差值的平方
                squared_difference = tf.square(difference)
                # 计算均方误差（MSE）
                loss = tf.reduce_mean(squared_difference)
                #loss = tf.reduce_mean(distance_tensor)
                grads = G_tape.gradient(loss, self.G_model.trainable_variables)
                # for var, grad in zip(self.G_model.trainable_variables, grads):
                #     if grad is not None:
                #         print(f'{var.name} gradient: {grad}')
                #     else:
                #         print(f'{var.name} gradient: None (可能是断开的梯度)')
                self.G_model.optimizer.apply_gradients(zip(grads, self.G_model.trainable_variables))
                self.loss.append(loss.numpy())
                # print('loss', loss.numpy())
                #print('distance_tensor', tf.reduce_mean(distance_tensor))
                # print('1 - normalized_mismatch_counts', tf.reduce_mean(normalized_mismatch_counts))
        else:
            agg_result= self.G_model(inputs)
        if step % 100 == 99 :
            self.update_target_network()
            self.G_model.save_weights('weight/GNN_weights.h5')
        agg_result = agg_result.numpy()

        return agg_result









