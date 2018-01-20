import gym
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from gym.envs.registration import registry, register, make, spec
from gym import spaces
def manhattan_distance(start, end):
    sx, sy = start
    ex, ey = end
    return abs(ex - sx) + abs(ey - sy)
class ZeppelinEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }
    def __init__(self):
        self.gravity = 9.8
        self.action_space = spaces.Discrete(5)
        self.width = 4
        self.height = 3
        start_pos = (0,0)
        end_pos = (self.width, self.height)
        self.grid = np.random.random_integers(0,25,(self.width,self.height))
        self.goal = end_pos
        self.state = start_pos
    def _render(self,mode,close):
        plt.imshow(self.grid, interpolation='none', cmap='gray')
        plt.show()

    def _step(self, action):
        reward = 1
        state = self.state
        if action == 0:
            next_state = state
        elif action == 1:# up
            next_state = (state[0],state[1]+1)
        elif action == 2:#left
            next_state = (state[0]-1,state[1])
        elif action == 3:#right
            next_state = (state[0]+1,state[1])
        elif action == 4:# down
            next_state = (state[0],state[1]-1)

        within_borders = next_state[0] >= 0 and next_state[1] >= 0  and next_state[0] <= self.width and next_state[1] <= self.height
        wind_speed = 90
        if within_borders:
            wind_speed = self.grid[next_state[0],next_state[1]]
            if wind_speed < 15:
                self.state = next_state
        distance = manhattan_distance(self.state, self.goal)
        reward = distance * int(wind_speed<15)
        print("distance: " + str(distance) + ", speed: " + str(wind_speed))
        done = False
        return self.state, reward, done, {}
    def _reset(self):
        print("do nothing")

register(
    id='zeppelin-v2',
    entry_point=__name__ +':ZeppelinEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

print(__name__)
if __name__ == '__main__':

  env = gym.make('zeppelin-v2')
  observation = env.reset()
  for i in range(10):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
#    print((observation, reward, done, info))
  env.render()
