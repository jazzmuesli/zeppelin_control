import numpy
import gym
import numpy as np
#import matplotlib as plt
import matplotlib.pyplot as plt
from gym.envs.registration import registry, register, make, spec
from gym import spaces
import pandas as pd

# vertical+horizontal distance
def manhattan_distance(start, end):
    sx, sy = start
    ex, ey = end
    return abs(ex - sx) + abs(ey - sy)


class ZeppelinEnv(gym.Env):
    metadata = {
        'render.modes': ['human']
    }
    def __init__(self):
        self.action_space = spaces.Discrete(5)
        #TODO: provide externally
        x = numpy.loadtxt(open("first_wind_prediction.txt", "rb"), delimiter=",")
        self.width = x.shape[0]
        self.height = x.shape[1]
        #TODO: provide externally
        citydata = pd.read_csv('CityData.csv')
        index = 1
        start_x = citydata['xid'][0]
        start_y = citydata['yid'][0]
        end_x = citydata['xid'][index]
        end_y = citydata['yid'][index]

        self.start_pos = (start_x, start_y)
        #(0,0)
        end_pos = (end_x,end_y)#(self.width, self.height)
        self.grid = x
        #np.random.random_integers(0,25,(self.width,self.height))
        self.goal = end_pos
        self.state = self.start_pos
        self.visited = []
        self.crash_wind_speed = 15


    def _render(self,mode,close):
        plt.imshow(self.grid.T, interpolation='none', cmap='hot')
        plt.plot(self.start_pos[0], self.start_pos[1], 'ro')
        plt.plot(self.goal[0], self.goal[1], 'ro')
        for item in self.visited:
            plt.plot(item[0], item[1], 'bo')

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

        within_borders = next_state[0] >= 0 and next_state[1] >= 0  and next_state[0] < self.width and next_state[1] < self.height
        wind_speed = 90
        if within_borders:
            wind_speed = self.grid[next_state[0],next_state[1]]
            if wind_speed < self.crash_wind_speed:
                self.state = next_state
        distance = manhattan_distance(self.state, self.goal)
        reward = distance * int(wind_speed<self.crash_wind_speed)
        #print("distance: " + str(distance) + ", speed: " + str(wind_speed))
        done = self.state == self.goal
        self.visited.append(self.state)
        return self.state, reward, done, {}
    def _reset(self):
        self.visited = []
        self.state = self.start_pos

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
