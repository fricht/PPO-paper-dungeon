import random


class AbstractEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = None
        return self.state

    def get_state(self):
        return self.state

    def step(self, action):
        self.state = None
        reward = None
        done = True
        return self.state, reward, done


class One2OneEnv(AbstractEnv):
    """
    This is bad. There is no value for time.
    """
    def __init__(self, n_args, episode_time=1000):
        super().__init__()
        self.args_count = n_args
        self.episode_time = episode_time
        self.current_time = 0
        self.reset()

    def new_state(self):
        self.state = [random.random() for _ in range(self.args_count)]

    def reset(self):
        self.new_state()
        return self.state

    def step(self, acton):
        reward = sum([1-abs(acton[i]-self.state[i]) for i in range(self.args_count)])
        self.new_state()
        self.current_time += 1
        done = False
        if self.current_time > self.episode_time:
            done = True
        return self.state, reward, done
