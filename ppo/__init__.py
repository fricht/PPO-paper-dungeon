import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self, output_neurons, activation_func=tf.keras.activations.sigmoid):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense2 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense3 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense_out = tf.keras.layers.Dense(output_neurons, activation=activation_func)

    def call(self, inputs):
        return self.dense_out(self.dense3(self.dense2(self.dense1(inputs))))


class PPO:
    def __init__(self):
        pass


if __name__ == "__main__":
    agent = Model(5)
    data = np.array([[1, 2, 3, 4, 5], [1, 3, 3, 4, 5]])
    print(data, agent(data))
