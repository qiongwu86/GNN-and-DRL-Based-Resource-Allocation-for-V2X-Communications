import tensorflow as tf
import tensorflow.keras as keras


class GraphModel(keras.Model):
    def __init__(self, sample_num=5, depth=2, dims=20, gcn=True, concat=True, dims_first_agg=[20],
                 dims_second_agg=[20], num_classes=20):
        super(GraphModel, self).__init__()
        self.sample_num = sample_num
        self.depth = depth
        self.dims = dims
        self.gcn = gcn
        self.concat = concat
        he_uniform_initializer = tf.keras.initializers.HeUniform()
        truncated_normal_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.01)

        # 一阶邻居聚合网络中的层
        self.dense_layers_first_agg = [
            tf.keras.layers.Dense(units=dims, activation=tf.nn.relu,  kernel_initializer=he_uniform_initializer)
            for dims in dims_first_agg
        ]

        # 如果depth为2，初始化二阶邻居聚合网络中的层
        if depth == 2:
            self.dense_layers_second_agg = [
                tf.keras.layers.Dense(units=dims, activation=tf.nn.relu, kernel_initializer=he_uniform_initializer)
                for dims in dims_second_agg
            ]

        # 多层分类器
        # self.classifier_layers = [
        #     tf.keras.layers.Dense(units=64, activation=tf.nn.leaky_relu, kernel_initializer=he_uniform_initializer),
        #     tf.keras.layers.Dense(num_classes, activation=None)
        # ]

    def aggregator(self, inputs, dense_layers, concat=True):
        node_features, neigh_features = inputs
        if concat:
            node_embed = tf.expand_dims(node_features, 1)
            to_feats = tf.concat([neigh_features, node_embed], 1)
        else:
            to_feats = neigh_features
        for layer in dense_layers:
            to_feats = layer(to_feats)
        combined = tf.reduce_mean(to_feats, axis=1)
        return combined

    def call(self, inputs):
        features, batch_nodes, s1_neighs, s2_neighs = inputs
        if self.depth == 1:
            node_fea = tf.nn.embedding_lookup(features, batch_nodes)
            neigh_1_fea = tf.nn.embedding_lookup(features, s1_neighs)
            tf.print('agg_node1', neigh_1_fea)
            agg_result = self.aggregator((node_fea, neigh_1_fea), self.dense_layers_first_agg, self.concat)
        else:
            node_fea = tf.nn.embedding_lookup(features, batch_nodes)
            neigh_1_fea = tf.nn.embedding_lookup(features, s1_neighs)
            #tf.print('agg_node1', neigh_1_fea)
            agg_node = self.aggregator((node_fea, neigh_1_fea), self.dense_layers_first_agg, self.concat)
            #tf.print('agg_node1', agg_node[0])
            neigh_2_fea = tf.nn.embedding_lookup(features, s2_neighs)
            agg_neigh1 = self.aggregator((neigh_1_fea, neigh_2_fea), self.dense_layers_first_agg, self.concat)
            agg_result = self.aggregator((agg_node, agg_neigh1), self.dense_layers_second_agg, self.concat)

        # # 通过分类器层序列传递agg_result
        # l2_norm = tf.norm(agg_result, ord=2, axis=1, keepdims=True)
        # # 进行L2归一化
        # normalized_output = agg_result / l2_norm
        # classifier_output = normalized_output
        # for layer in self.classifier_layers:
        #     classifier_output = layer(classifier_output)

        return agg_result