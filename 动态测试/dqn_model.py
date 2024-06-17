import tensorflow as tf


class DQNModel:
    def __init__(self):
        self.learning_rate = 0.01
        self.learning_rate_minimum = 0.0001
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 500000
        self.initialize_network()
        self.compile_model()

    def initialize_network(self):
        self.model = self.build_dqn()
        self.target_model = self.build_dqn()

    def build_dqn(self):
        n_input = 102
        n_output = 60

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_input,)),
            tf.keras.layers.Dense(500, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(250, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(120, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1)),
            tf.keras.layers.Dense(n_output, activation='relu',
                                  kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.1))
        ])

        return model

    def forward(self, inputs):
        inputs = tf.reshape(inputs, [-1, 102])
        return self.model(inputs, training=True)

    def forward_target(self, inputs):
        return self.target_model(inputs, training=False)

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def clipped_error(self, x):
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

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
            step_lr = lr_schedule(self.model.optimizer.iterations)
            return tf.maximum(step_lr, self.learning_rate_minimum)

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=minimum_lr_fn, rho=0.95, epsilon=0.01)

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        # self.model.build(input_shape=(None, 102))
        # self.model.load_weights('weight/dqn_weights.h5')

    @tf.function
    def train_step(self, inputs, targets, actions):
        with tf.GradientTape() as tape:
            q_values = self.forward(inputs)
            action_masks = tf.one_hot(actions, q_values.shape[1])
            q_acted = tf.reduce_sum(q_values * action_masks, axis=1)
            loss = tf.reduce_mean(tf.square(targets - q_acted))

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, q_values
