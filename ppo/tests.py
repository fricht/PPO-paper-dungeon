import __init__ as ppo
import unittest
import random


class TestEnv(ppo.Env):
    def __init__(self):
        self.state = None
        self.i = None

    def reset(self):
        self.i = 0
        self.state = ppo.np.array([0]*10)
        self.state[5] = 1
        return self.state

    def step(self, action):
        self.i += 1
        reward = 10 * abs(ppo.np.argmax(action) - ppo.np.argmax(self.state))
        done = False
        self.state = ppo.np.array([0]*10)
        self.state[random.randint(0, 9)] = 1
        if self.i > 1000:
            done = True
        return self.state, reward, done


class TestPPO(unittest.TestCase):
    def test_all(self):
        PPO_agent = ppo.PPO(10, 10)
        # testing
        env = TestEnv()
        state = env.reset()
        action = PPO_agent.get_action(state)
        next_state, reward, _ = env.step(action)
        print(state, action, reward)
        # training
        data = PPO_agent.generate_episodes(TestEnv(), random.randint(8, 30))
        PPO_agent.train(*data, batch_size=random.randint(24, 64))
        # testing
        env = TestEnv()
        state = env.reset()
        action = PPO_agent.get_action(state)
        next_state, reward, _ = env.step(action)
        print(state, action, reward)


class TestAgents(unittest.TestCase):
    def test_actor(self):
        n_out = random.randint(2, 20)
        actor = ppo.Actor(n_out)
        in_data = ppo.np.random.normal(size=(random.randint(2, 20), random.randint(2, 20)))
        out = actor(in_data)
        self.assertIsInstance(out, ppo.tf.Tensor, "Wtf does the actor network outputs ???")
        self.assertEqual(out.shape, ppo.tf.TensorShape((in_data.shape[0], n_out)), "Weird output shape of the actor model")

    def test_critic(self):
        critic = ppo.Critic()
        in_data = ppo.np.random.normal(size=(random.randint(2, 20), random.randint(2, 20)))
        out = critic(in_data)
        self.assertIsInstance(out, ppo.tf.Tensor, "Wtf does the critic network outputs ???")
        self.assertEqual(out.shape, ppo.tf.TensorShape((in_data.shape[0], 1)), "Weird output shape of the critic model")


if __name__ == "__main__":
    print("\nTesting ...\n")
    unittest.main()
