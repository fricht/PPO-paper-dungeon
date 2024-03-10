# TODO : test this class
class Env:
    def __init__(self):
        self.state = None

    def reset(self):
        self.state = None
        return self.state

    def step(self, action):
        self.state = None
        reward = None
        done = True
        return self.state, reward, done
