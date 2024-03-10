import tensorflow as tf
import numpy as np
import env


class Model(tf.keras.Model):
    def __init__(self, output_neurons, activation_func=tf.nn.sigmoid):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense2 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense3 = tf.keras.layers.Dense(128, activation=activation_func)
        self.dense_out = tf.keras.layers.Dense(output_neurons, activation=activation_func)

    def call(self, inputs):
        return self.dense_out(self.dense3(self.dense2(self.dense1(inputs))))


class PPO:
    def __init__(self, actions_neurons, gamma=0.95):
        self.gamma = gamma
        self.actions_neurons = actions_neurons
        self.actor = Model(actions_neurons)
        self.critic = Model(1, activation_func=None)
        self.actor.compile(loss=tf.keras.losses.MeanSquaredError, optimizer='adam')

    def compute_value_function(self, rewards):
        returns = []
        discounted = 0
        for r in reversed(rewards):
            discounted = r + self.gamma * discounted
            returns.insert(0, discounted)
        return returns

    def train(self, env, epochs=100, episodes=10, batch_size=32, sub_epochs=10, max_iters_episode=1000):
        for i in range(epochs):
            print(i / epochs)
            # actor perform in env
            total_states = []
            total_actions = []
            total_rewards = []
            total_value = []
            for _ in range(episodes):
                states = [env.reset()]
                actions = []
                rewards = []
                for j in range(max_iters_episode):
                    action = None
                    state, reward, done = env.step(action)
                    actions.append(action)
                    rewards.append(reward)
                    if done:
                        break
                    states.append(state)
                values = self.compute_value_function(rewards)
                total_states += states
                total_actions += actions
                total_rewards += rewards
                total_value += values
            # train critic
            self.critic.fit(x=np.array([total_states, total_actions]).transpose(), y=np.array([total_value]).transpose(), batch_size=batch_size, epochs=sub_epochs)
            # eval actions weights
            weighted_best = []
            for state in states:
                weighted_best.append()
            weights = self.critic.predict(np.identity(self.state_size)).numpy().flatten()
            #


if __name__ == "__main__":
    agent = Model(5)
    data = np.array([[1, 2, 3, 4, 5], [1, 3, 3, 4, 5]])
    print(data, agent(data))
