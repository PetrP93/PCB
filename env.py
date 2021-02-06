from matplotlib import pyplot as plt
import numpy as np

plt.ion()


class PCB_Env:

    def __init__(self, size_h, size_w):
        self.field = None
        self.agent_position = None
        self.target_position = None
        self.size_h = size_h
        self.size_w = size_w
        self.action_space = {'right': self.step_right, 'left': self.step_left,
                             'up': self.step_up, 'down': self.step_down}
        self.standing_still = 0

    def check(self, x, y):
        grid_check = self.size_w > x >= 0 and self.size_h > y >= 0
        cross_check = self.field[y][x] == 0
        return grid_check and cross_check

    def step_up(self):
        x, y = self.agent_position
        if self.check(x, y+1):
            self.field[y+1][x] = 1
            self.agent_position = (x, y+1)
            return True
        else:
            return False

    def step_down(self):
        x, y = self.agent_position
        if self.check(x, y-1):
            self.field[y-1][x] = 1
            self.agent_position = (x, y-1)
            return True
        else:
            return False

    def step_left(self):
        x, y = self.agent_position
        if self.check(x-1, y):
            self.field[y][x-1] = 1
            self.agent_position = (x-1, y)
            return True
        else:
            return False

    def step_right(self):
        x, y = self.agent_position
        if self.check(x+1, y):
            self.field[y][x+1] = 1
            self.agent_position = (x+1, y)
            return True
        else:
            return False

    def step(self, action):
        assert action in self.action_space
        step_function = self.action_space[action]
        result = step_function()
        done = self.agent_position == self.target_position
        if not result:
            reward = -10
            self.standing_still += 1
        else:
            reward = 0
            self.standing_still = 0
        if done:
            reward = 10100 - self.field.sum()
        return self.field, reward, done, self.standing_still

    def reset(self, start_x, start_y, end_x, end_y):
        self.field = np.zeros((self.size_h, self.size_w))
        self.field[start_y][start_x] = 1
        self.field[end_y][end_x] = 2
        self.agent_position = (start_x, start_y)
        self.target_position = (end_x, end_y)
        self.standing_still = 0
        return self.field

    def render(self):
        plt.imshow(self.field)
        plt.draw()
        plt.pause(0.0001)

