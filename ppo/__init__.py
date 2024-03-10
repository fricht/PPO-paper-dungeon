import tensorflow as tf
import numpy as np


class Actor(tf.keras.Model):
    def __init__(self, actions_count):
        super().__init__()
        self.l1 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.l2 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.l3 = tf.keras.layers.Dense(128, activation=tf.nn.sigmoid)
        self.lout = tf.keras.layers.Dense(actions_count, activation=tf.nn.sigmoid)

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
    # TODO : write unit tests for this class
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
        return tf.Tensor(returns)

    # thank you chat GPT, i don't understand everything
    def train_step(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape_actor, tf.GradientTape() as tape_critic:
            probs = self.actor(states)
            values = self.critic(states)
            action_masks = tf.one_hot(actions, probs.shape[1])

            advantages = advantages - tf.reduce_mean(advantages)
            actor_loss = -tf.reduce_sum(tf.math.log(tf.reduce_sum(probs * action_masks, axis=1)) * advantages)

            critic_loss = tf.reduce_sum(tf.square(returns - values))

            total_loss = actor_loss + critic_loss

        grads_actor = tape_actor.gradient(total_loss, self.actor.trainable_variables)
        grads_critic = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_opti.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.critic_opti.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

    # i understood better this function
    def train_episode(self, states, actions, rewards, batch_size=32):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        values = self.critic(states)
        advantages = rewards - values.numpy().flatten()

        returns = self.compute_returns(rewards)

        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        for _ in range(10):  # Number of PPO epochs
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)

                self.train_step(batch_states, batch_actions, batch_advantages, batch_returns)
