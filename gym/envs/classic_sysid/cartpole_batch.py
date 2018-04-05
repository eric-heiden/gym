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
from copy import deepcopy

logger = logging.getLogger(__name__)
class CartPoleEnvSysIDBatch(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.N = 32
        self.state = np.zeros((self.N, 8))
        self.fixedlen = True

        # initialize the system identification parameters.
        def expand(x):
            ratio = 4.0
            return [x/ratio, x*ratio]
        self.sysid_params = OrderedDict([
            #('masscart',  [expand(1.0), 0]),
            #('masspole',  [expand(0.1), 0]),
            #('length',    [expand(0.5), 0]),
            #('force_mag', [expand(10.0), 0]),
            ('masscart',  [[1, 1], 0]),
            ('masspole',  [[0.1, 0.1], 0]),
            ('length',    [[0.5, 0.5], 0]),
            ('force_mag', [[10, 10], 0]),
        ])
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.length = 0.5 # actually half the pole's length
        # self.force_mag = 10.0

        low_sysid = np.array([p[0][0] for p in self.sysid_params.values()])
        high_sysid = np.array([p[0][1] for p in self.sysid_params.values()])

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
        self.steps_beyond_done = None

        self.ep_len = 512
        self.tick = 0

        self._seed()
        self.sample_sysid()

    def sample_sysid(self):
        print("sampling new sysid")
        for p in self.sysid_params.values():
            p[1] = self.np_random.uniform(p[0][0], p[0][1], (self.N))
        self.masscart = self.sysid_params['masscart'][1]
        self.masspole = self.sysid_params['masspole'][1]
        self.total_mass = (self.masspole + self.masscart)
        self.length = self.sysid_params['length'][1]
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = self.sysid_params['force_mag'][1]
        self.state[:,4:] = self.sysid_values()

    def sysid_values(self):
        v = np.array([p[1] for p in self.sysid_params.values()]).T
        assert v.shape == (self.N, self.sysid_dim)
        return v

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert action.shape == (self.N,1)
        action = action.flatten()
        action[action > 1] = 1
        action[action < -1] = -1
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # N * 4
        state = self.state
        x, x_dot, theta, theta_dot, *sysid_states = (np.squeeze(a) for a in np.split(self.state, 4 + self.sysid_dim, axis=1))
        force = self.force_mag * self.total_mass * action
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        damping = 0.998 # lose 10% of velocity in one second
        x  = x + self.tau * x_dot
        x_dot = damping * x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = damping*theta_dot + self.tau * thetaacc
        self.state = np.stack([x, x_dot, theta, theta_dot] + sysid_states, axis=1)

        theta_wrap = (theta + np.pi) % (2 * np.pi) - np.pi

        if self.fixedlen:
            #print("x = {}, xdot = {}, th = {}, thdot = {}".format(
                #x[0], x_dot[0], theta[0], theta_dot[0]))
            vel_rew = - 0.05 * np.abs(theta_dot) - 0.02 * np.abs(x_dot)
            reward = -(1.0/np.pi) * np.abs(theta_wrap) - 0.05*np.abs(x) + vel_rew
            reward[np.abs(x) > self.x_threshold] -= 100

            self.tick += 1
            self.done = x != x
            if self.tick >= self.ep_len:
                self.done = x == x
                self.reset_done()
                self.tick = 0
        else:
            self.done = np.logical_or(
                np.abs(x) > self.x_threshold,
                np.abs(theta) > 8 * np.pi
            )
            theta_wrap = (theta + np.pi) % (2 * np.pi) - np.pi
            assert all(np.abs(theta_wrap) <= np.pi)

            #reward = 0.1 - 0.1*np.abs(x) - 0.1*np.abs(theta_wrap) - 0.01*np.abs(x_dot) - 0.01*np.abs(theta_dot)
            #reward = 0.1 * reward
            #reward = 1.0 * (np.abs(theta) < 0.1) + 0.1 * (np.abs(theta < np.pi / 2))
            reward = -0.1 * np.abs(theta_wrap)
            reward[self.done] = -100
            self.reset_done()

        self.reward = reward

        # after self.reset_done(), all states that were done
        # have already been reset to new states
        return self.state, reward, self.done, {}

    def reset_done(self):
        prev_states = deepcopy(self.state)
        new_states = self._sample_reset()
        self.state[self.done,:] = new_states[self.done,:]
        assert np.all(self.state[:,4:] == prev_states[:,4:])

    def _sample_reset(self):
        x = self.x_threshold / 115.0
        #t = self.theta_threshold_radians / 110.0
        th = np.pi

        state = np.concatenate([
            self.np_random.uniform(low=[-x, -x, th, 0], high=[x, x, th, 0], size=(self.N, 4)),
            self.sysid_values()], 1)
        assert state.shape == (self.N, self.sysid_dim + 4)
        return state

    def _reset(self):
        self.state = self._sample_reset()
        return self.state

    def _render(self, mode='human', close=False):
        #return
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
            self.pole = pole

        if self.state is None: return None

        color_good = np.array([0.1, 0.8, 0.2])
        color_bad = np.array([0.9, 0.1, 0.15])
        r = -max(-1.0, self.reward[0])
        self.pole.set_color(*(r * color_bad + (1 - r) * color_good))

        x = self.state[0]
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
