import tensorflow as tf
import numpy as np

class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_size, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(tf.keras.Model):
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)

class PPOAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=0.2, lr_actor=1e-4, lr_critic=2e-4):
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        probs = self.actor(state)
        action = np.random.choice(range(probs.shape[1]), p=probs.numpy()[0])
        return action

    def compute_returns(self, rewards):
        returns = []
        discounted_sum = 0
        for r in reversed(rewards):
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        return returns

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

        self.optimizer_actor.apply_gradients(zip(grads_actor, self.actor.trainable_variables))
        self.optimizer_critic.apply_gradients(zip(grads_critic, self.critic.trainable_variables))

    def train(self, states, actions, rewards):
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

            batch_size = 32
            for start in range(0, len(states), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch_states = tf.gather(states, batch_indices)
                batch_actions = tf.gather(actions, batch_indices)
                batch_advantages = tf.gather(advantages, batch_indices)
                batch_returns = tf.gather(returns, batch_indices)

                self.train_step(batch_states, batch_actions, batch_advantages, batch_returns)

# Example usage:
state_size = 4
action_size = 2

ppo_agent = PPOAgent(state_size, action_size)

# Assuming you have a function to generate your environment and collect data
# states, actions, rewards = generate_episodes(env, ppo_agent)

# Train the agent
ppo_agent.train(states, actions, rewards)
