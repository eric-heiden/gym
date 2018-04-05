"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from collections import OrderedDict

logger = logging.getLogger(__name__)
class CartPoleEnvSysID(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        # initialize the system identification parameters.
        self.sysid_params = OrderedDict([
            #('masscart',  [[0.5, 2], 0]),
            #('masspole',  [[0.05, 0.2], 0]),
            #('length',    [[0.25, 1], 0]),
            #('force_mag', [[4, 15], 0]),
            ('masscart',  [[1, 1], 0]),
            ('masspole',  [[0.1, 0.1], 0]),
            ('length',    [[0.5, 0.5], 0]),
            ('force_mag', [[20, 20], 0]),
        ])
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.length = 0.5 # actually half the pole's length
        # self.force_mag = 10.0

        low_sysid = np.array([p[0][0] for p in self.sysid_params.values()])
        high_sysid = np.array([p[0][1] for p in self.sysid_params.values()])
        self._seed()
        self.sample_sysid()

        self.gravity = 9.8
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 45 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # observe: x, xdot, th, thdot
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        low = np.concatenate([-high, low_sysid])
        high = np.concatenate([high, high_sysid])
        self.observation_space = spaces.Box(low, high)
        self.sysid_dim = len(self.sysid_params)

        # act: between -1 and 1, will be scaled by force_mag.
        # extra push for policy to learn SysID of force_mag.
        self.action_space = spaces.Box(np.array([-1]), np.array([1]))

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def sample_sysid(self):
        for p in self.sysid_params.values():
            p[1] = self.np_random.uniform(*p[0])
        self.masscart = self.sysid_params['masscart'][1]
        self.masspole = self.sysid_params['masspole'][1]
        self.total_mass = (self.masspole + self.masscart)
        self.length = self.sysid_params['length'][1]
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = self.sysid_params['force_mag'][1]

    def sysid_values(self):
        return 0 * np.array([p[1] for p in self.sysid_params.values()])

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        action[action > 1] = 1
        action[action < -1] = -1
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot, *_ = state
        force = self.force_mag * self.total_mass * action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.concatenate([np.array([x, x_dot, theta, theta_dot]), self.sysid_values()])
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            #reward = 1.0 - 0.1 * np.sum(np.abs(self.state[:4]))
            reward = 1.0 - 0.1*abs(x) - 0.1*abs(theta) - 0.01*abs(x_dot) - 0.01*abs(theta_dot)
            reward = reward[0]
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -1000
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        reward = 0.1 * reward
        return self.state, reward, done, {}

    def _reset(self):
        x = self.x_threshold / 115.0
        t = self.theta_threshold_radians / 110.0

        self.state = np.concatenate([
            self.np_random.uniform(low=[-x, -x, -t, -t], high=[x, x, t, t]),
            self.sysid_values()])
        self.steps_beyond_done = None
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        return
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
