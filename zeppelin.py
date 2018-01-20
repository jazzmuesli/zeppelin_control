import gym
import numpy as np
import matplotlib as plt
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
        width = 4
        height = 3
        start_pos = (0,0)
        end_pos = (width, height)
        self.grid = np.random.random_integers(0,25,(width,height))
        self.goal = end_pos
        self.state = start_pos
    def _render(self,mode,close):
        #self.grid[self.goal[1]][self.goal[0]] = 1
#        plt.figure(0)
        #plt.clf()
        plt.imshow(self.grid, interpolation='none', cmap='gray')
#        plt.imshow(costSurfaceArray, cmap='hot', interpolation='nearest')
        #plt.savefig(path + "maze.png")

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
        wind_speed = self.grid[next_state[0],next_state[1]]
        if next_state[0] >= 0 and next_state[1] >= 0 and wind_speed < 15:
            self.state = next_state
        distance = manhattan_distance(state, self.goal)
        reward = distance * int(wind_speed<15)
        print("distance: " + str(distance) + ", speed: " + str(wind_speed))
        done = False
        return self.state, reward, done, {}
    def _reset(self):
        print("do nothing")


if __name__ == 'main':
  env = gym.make('zeppelin-v2')
  observation = env.reset()
  for i in range(10):
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
#    print((observation, reward, done, info))
  env.render()
