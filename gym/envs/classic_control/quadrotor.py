"""
3D quadrotor environment.
"""

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering3d as r3d
import numpy as np
from numpy.linalg import norm
import traceback
import sys
import csv
import datetime
from copy import deepcopy

logger = logging.getLogger(__name__)

GRAV = 9.81

def quadrotor_3dmodel(diam):
    r = diam / 2
    prop_r = 0.3 * diam
    prop_h = prop_r / 15.0

    # "X" propeller configuration, start fwd left, go clockwise
    rr = r * np.sqrt(2)/2
    deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
    colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
    def disc(translation, color):
        color = 0.3 * np.array(list(color)) + 0.2
        disc = r3d.transform_and_color(r3d.translate(translation), color,
            r3d.cylinder(prop_r, prop_h, 32))
        return disc
    props = [disc(d, c) for d, c in zip(deltas, colors)]

    arm_thicc = diam / 20.0
    arm_color = (0.5, 0.5, 0.5)
    arms = r3d.transform_and_color(
        np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
        [r3d.box(diam/10, diam, arm_thicc), r3d.box(diam, diam/10, arm_thicc)])

    arrow = r3d.Color((0.3, 0.3, 1.0), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))

    bodies = props + [arms, arrow]
    return r3d.Transform(np.eye(4), bodies)

def is_orthonormal(m):
    return np.max(np.abs(np.matmul(m, m.T) - np.eye(3)).flatten()) < 0.00001

def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

def norm2(x):
    return np.sum(x ** 2)

def orth_cols(m):
    u, s, v = np.linalg.svd(m)
    return np.matmul(u, v)

def rand_uniform_rot3d(np_random):
    randunit = lambda: normalize(np_random.normal(size=(3,)))[0]
    up = randunit()
    fwd = randunit()
    while np.dot(fwd, up) > 0.95:
        fwd = randunit()
    left = normalize(np.cross(up, fwd))
    up = np.cross(fwd, left)
    rot = np.hstack([fwd, left, up])

def npa(*args):
    return np.array(args)

def omega_mat_deriv(omega):
    x, y, z = omega
    dRdt = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return dRdt

def hinge_loss(x, loss_above):
    try:
        return np.max(0, x - loss_above)
    except TypeError:
        return max(0, x - loss_above)

class QuadrotorDynamics(object):
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    def __init__(self, mass, arm_length, inertia, thrust_to_weight=2.0, torque_to_thrust=0.05):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust = GRAV * mass * thrust_to_weight / 4.0
        self.torque = torque_to_thrust * self.thrust
        scl = arm_length / norm([1,1,0])
        self.prop_pos = scl * np.array([
            [1,  1, -1, -1],
            [1, -1, -1,  1],
            [0,  0,  0,  0]]).T # row-wise easier with np
        self.prop_ccw = np.array([1, -1, 1, -1])

    # pos, vel, in world coords
    # rotation is (body coords) -> (world coords)
    # omega in body coords
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3,3)
        assert is_orthonormal(rotation)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega)
        self.thrusts = deepcopy(thrusts)
        self.crashed = False

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, np_random, box, vel_max=15.0, omega_max=2*np.pi):
        pos = np_random.uniform(low=-box, high=box, size=(3,))
        vel = np_random.uniform(low=-vel_max, high=vel_max, size=(3,))
        omega = np_random.uniform(low=-omega_max, high=omega_max, size=(3,))
        rot = rand_uniform_rot3d(np_random)
        self.set_state(pos, vel, rot, omega)

    def step(self, thrust_cmds, dt):

        if self.pos[2] <= self.arm:
            # crashed, episode over
            self.pos[2] = self.arm
            self.vel *= 0
            self.omega *= 0
            self.crashed = True
            return

        assert np.all(thrust_cmds >= 0)
        assert np.all(thrust_cmds <= 1)
        thrusts = self.thrust * thrust_cmds
        thrust = npa(0,0,np.sum(thrusts))
        torques = np.cross(self.prop_pos, [0, 0, 1]) * thrusts[:,None]
        torques[:,2] += self.torque * self.prop_ccw * thrust_cmds
        torque = np.sum(torques, axis=0)

        # TODO add noise

        vel_damp = 0.99
        omega_damp = 0.99

        # rotational dynamics
        omega_dot = ((1.0 / self.inertia) *
            (np.cross(-self.omega, self.inertia * self.omega) + torque))
        self.omega = omega_damp * self.omega + dt * omega_dot

        dRdt = np.matmul(omega_mat_deriv(self.omega), self.rot)
        rot = self.rot + dt * dRdt;
        self.rot = orth_cols(rot)

        # translational dynamics
        acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, thrust)
        self.vel = vel_damp * self.vel + dt * acc
        self.pos = self.pos + dt * self.vel

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

class ChaseCamera(object):
    def __init__(self, pos=npa(0,0,0), vel=npa(0,0,0)):
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(np.cross(vel, npa(0, 0, 1)))
        self.view_dist = 4

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(-self.pos_smooth)
        if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(np.cross(ideal_vel, up))
        else:
            right, _ = normalize(np.cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(np.cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        mass = 0.5
        arm_length = 0.33 / 2.0
        inertia = mass * npa(0.01, 0.01, 0.02)
        self.thrust_to_weight = 2.0
        self.dynamics = QuadrotorDynamics(mass, arm_length, inertia,
            thrust_to_weight=self.thrust_to_weight)

        self.ep_len = 256
        self.tick = 0
        self.dt = 1.0 / 50.0

        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (self.thrust_to_weight - 1.0) * np.ones(4)
        self.action_space = spaces.Box(low, high)
        print("action space:", low, high)
        # pos, vel, rot, omega
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim)
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        csvpath = './loss_log' + str(datetime.datetime.now()) + '.csv'
        self.csvwriter = csv.writer(open(csvpath, 'w', newline=''))
        row = ['pos_smooth', 'goal', 'pos', 'alt', 'vel', 'omega', 'action', 'crash']
        self.loss_integral = np.zeros(len(row) - 1)
        self.csvwriter.writerow(row)

        self.n_resets = -1
        self.lifetime_pos_smooth = 0

        self.box = 1.0

        self._seed()
        self.reset()
        self.viewer = None
        self.model = None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # noop for gfx testing
        #return self.dynamics.state_vector(), 0, False, {}
        action = (action + 1.0) / self.thrust_to_weight
        #action = action * np.ones(4)
        action[action < 0] = 0
        action[action > 1] = 1
        #action[:] = 0.5
        #action = npa(0.6, 0.4, 0.6, 0.4)
        self.dynamics.step(action, self.dt)
        self.camera.step(self.dynamics.pos, self.dynamics.vel)

        self.tick += 1
        done = self.tick >= self.ep_len

        vel = self.dynamics.vel
        to_goal = -self.dynamics.pos

        # note we don't want to penalize distance^2 because in harder instances
        # the initial distance can be very far away
        loss_pos = norm(to_goal)

        # penalize velocity away from goal but not towards
        loss_vel_away = 0.1 * (norm(vel) * norm(to_goal) - np.dot(vel, to_goal))

        # penalize altitude above this threshold
        max_alt = 3.0
        loss_alt = 2 * hinge_loss(self.dynamics.pos[2], 3) ** 2

        loss_spin = 0.02 * norm2([1, 1, 10] * self.dynamics.omega)
        loss_crash = 50 * self.dynamics.crashed
        loss_battery = 0.02 * norm2(action)

        goal_thresh = 1.0 # within this distance, start rewarding
        goal_max = 0 # max reward when exactly at goal
        a = -goal_max / (goal_thresh**2)
        reward_goal = max(0,  a * norm2(to_goal) + goal_max)

        #row = ['goal', 'pos', 'alt', 'vel', 'omega', 'action', 'crash']
        self.loss_integral += self.dt * np.array([-reward_goal, loss_pos, loss_alt,
            loss_vel_away, loss_spin, loss_battery, loss_crash])

        if self.tick % 30 == 0 and False:
            print("  losses:")
            print("    pos =", loss_pos)
            print("    alt =", loss_alt)
            print("   spin =", loss_spin)
            print("  crash =", loss_crash)
            print("battery =", loss_battery)
            print("   goal =", -reward_goal)
        reward = -self.dt * np.sum([
            -reward_goal,
            loss_pos, loss_vel_away, loss_alt, loss_spin, loss_crash, loss_battery])

        sv = self.dynamics.state_vector()
        return sv, reward, done, {}

    def _reset(self):
        alpha = 0.999
        self.n_resets += 1
        self.lifetime_pos_smooth *= alpha
        self.lifetime_pos_smooth += (1.0 - alpha) * self.loss_integral[0]

        x, y = self.np_random.uniform(-self.box, self.box, size=(2,))
        if self.box < 20:
            self.box *= 1.0003 # x20 after 10000 resets
        print("box:", self.box)
        #x, y = 20, 0
        z = self.np_random.uniform(1, 3)
        pos = npa(x, y, z)
        #pos = npa(0, 0, 2)
        vel = omega = npa(0, 0, 0)
        #rotz = np.random.uniform(-np.pi, np.pi)
        #rotation = r3d.rotz(rotz)
        #rotation = rotation[:3,:3]
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.camera = ChaseCamera(pos=pos, vel=vel)
        self.goal = npa(0, 0, 2)
        self.tick = 0
        np.set_printoptions(precision=4)
        if np.any(self.loss_integral != 0):
            self.csvwriter.writerow(
                [self.lifetime_pos_smooth] + list(self.loss_integral))
        self.loss_integral *= 0
        return self.dynamics.state_vector()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 400

        if self.viewer is None:
            self.viewer = r3d.Viewer(screen_width, screen_height)
            diameter = 2 * self.dynamics.arm

            self.quad_transform = quadrotor_3dmodel(diameter)

            self.shadow_transform = r3d.transform_and_color(
                np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

            floor = r3d.CheckerTexture(
                r3d.rect((1000, 1000), (0, 100), (0, 100)))

            goal = r3d.transform_and_color(r3d.translate(self.goal),
                (0.5, 0.4, 0), r3d.sphere(diameter/2, 18))

            #world = r3d.World([
               #self.quad_transform, self.shadow_transform, floor, goal])
            world = r3d.BackToFront([
                floor, self.shadow_transform, goal, self.quad_transform])
            batch = r3d.Batch()
            world.build(batch, None)

            self.viewer.add_batch(batch)

        matrix = r3d.trans_and_rot(self.dynamics.pos, self.dynamics.rot)
        self.quad_transform.set_transform(matrix)

        shadow_pos = 0 + self.dynamics.pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform(matrix)

        eye, center, up = self.camera.look_at()
        self.viewer.look_at(eye, center, up)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
