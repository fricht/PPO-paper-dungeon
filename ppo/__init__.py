import tensorflow as tf
import numpy as np


class Actor(tf.keras.Model):
    def __init__(self, actions_count):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.l2 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.l3 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.lout = tf.keras.layers.Dense(out_count, activation=tf.nn.sigmoid)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return self.lout(x)


class Critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128)
        self.l2 = tf.keras.layers.Dense(128)
        self.l3 = tf.keras.layers.Dense(128)
        self.lout = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        return self.lout(x)


class PPO:
    def __init__(self, actions_count, gamma=0.95, actor_lr=0.001, critic_lr=0.001):
        self.actor = Actor(actions_count)
        self.critic = Critic()
        self.actor_opti = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_opti = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.gamma = gamma # the discount factor

    def get_action(self, state):
        return np.argmax(self.actor([state])[0])

    def compute_discounted_returns(self, rewards):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns
