import __init__ as ppo
import unittest
import random


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
