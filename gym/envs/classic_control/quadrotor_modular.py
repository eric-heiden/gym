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

def is_orthonormal(m):
    return np.max(np.abs(np.matmul(m, m.T) - np.eye(3)).flatten()) < 0.00001

def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

def norm2(x):
    return np.sum(x ** 2)

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
        self.thrust_to_weight = thrust_to_weight
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
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
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

        x, y, z = self.omega
        omega_mat_deriv = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        dRdt = np.matmul(omega_mat_deriv, self.rot)
        # update and orthogonalize
        u, s, v = np.linalg.svd(self.rot + dt * dRdt)
        self.rot = np.matmul(u, v)

        # translational dynamics
        acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, thrust)
        self.acc = acc
        self.vel = vel_damp * self.vel + dt * acc
        self.pos = self.pos + dt * self.vel

        self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, GRAV])

    # return eye, center, up suitable for gluLookAt representing onboard camera
    def look_at(self):
        degrees_down = 45.0
        R = self.rot
        # camera slightly below COM
        eye = self.pos + np.matmul(R, [0, 0, -0.02])
        theta = np.radians(degrees_down)
        to, _ = normalize(np.cos(theta) * R[:,0] - np.sin(theta) * R[:,2])
        center = eye + to
        up = np.cross(to, R[:,1])
        return eye, center, up

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])


def default_dynamics():
    # similar to AscTec Hummingbird
    # TODO: dictionary of dynamics of real quadrotors
    mass = 0.5
    arm_length = 0.33 / 2.0
    inertia = mass * npa(0.01, 0.01, 0.02)
    thrust_to_weight = 2.0
    return QuadrotorDynamics(mass, arm_length, inertia,
        thrust_to_weight=thrust_to_weight)

# different control schemes.

# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
class ShiftedMotorControl(object):
    def __init__(self, dynamics):
        pass

    def action_space(self, dynamics):
        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        return spaces.Box(low, high)

    # dynamics passed by reference
    def step(self, dynamics, action, dt):
        action = (action + 1.0) / dynamics.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        dynamics.step(action, dt)

# TODO:
# class AttitudeRateControl
# class AttitudeControl
# class VelocityControl

def goal_seeking_reward(dynamics, goal, action, dt):
    vel = dynamics.vel
    to_goal = -dynamics.pos

    # note we don't want to penalize distance^2 because in harder instances
    # the initial distance can be very far away
    loss_pos = norm(to_goal)

    # penalize velocity away from goal but not towards
    # TODO this is too hacky, try to not use it
    loss_vel_away = 0.1 * (norm(vel) * norm(to_goal) - np.dot(vel, to_goal))

    # penalize altitude above this threshold
    max_alt = 3.0
    loss_alt = 2 * hinge_loss(dynamics.pos[2], 3) ** 2

    # penalize yaw spin more
    loss_spin = 0.02 * norm2([1, 1, 10] * dynamics.omega)

    loss_crash = 50 * dynamics.crashed

    loss_effort = 0.02 * norm2(action)

    # TODO this is too hacky, try not to use it
    goal_thresh = 1.0 # within this distance, start rewarding
    goal_max = 0 # max reward when exactly at goal
    a = -goal_max / (goal_thresh**2)
    reward_goal = max(0,  a * norm2(to_goal) + goal_max)

    reward = -dt * np.sum([
        -reward_goal,
        loss_pos, loss_vel_away, loss_alt, loss_spin, loss_crash, loss_effort])

    return reward


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


class Quadrotor3DScene(object):
    def __init__(self, goal, dynamics, w, h, resizable, visible=True):
        self.viewer = r3d.Viewer(w, h, resizable=resizable, visible=visible)
        self.chase_cam = ChaseCamera(dynamics.pos, dynamics.vel)

        diameter = 2 * dynamics.arm
        self.quad_transform = self._quadrotor_3dmodel(diameter)

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

        floor = r3d.ProceduralTexture(r3d.TEX_CHECKER, (0.15, 0.25),
            r3d.rect((1000, 1000), (0, 100), (0, 100)))

        goal = r3d.transform_and_color(r3d.translate(goal),
            (0.5, 0.4, 0), r3d.sphere(diameter/2, 18))

        world = r3d.BackToFront([
            floor, self.shadow_transform, goal, self.quad_transform])
        batch = r3d.Batch()
        world.build(batch, None)
        self.viewer.add_batch(batch)

    def _quadrotor_3dmodel(self, diam):
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

    def update_state(self, dynamics):
        matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
        self.quad_transform.set_transform(matrix)
        self.firstperson_lookat = dynamics.look_at()
        shadow_pos = 0 + dynamics.pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform(matrix)

    def render_chase(self, return_rgb_array):
        eye, center, up = self.chase_cam.look_at()
        self.viewer.set_fov(45)
        self.viewer.look_at(eye, center, up)
        return self.viewer.render(return_rgb_array=return_rgb_array)

    def render_firstperson(self, return_rgb_array):
        self.viewer.set_fov(90)
        self.viewer.look_at(*self.firstperson_lookat)
        return self.viewer.render(return_rgb_array=return_rgb_array)


class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        self.controller = ShiftedMotorControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene = None

        # pos, vel, rot, omega
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim)
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        # TODO get this from a wrapper
        self.ep_len = 10000
        self.tick = 0
        self.dt = 1.0 / 50.0

        self._seed()

        # size of the box from which initial position will be randomly sampled
        # grows a little with each episode
        self.box = 1.0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.controller.step(self.dynamics, action, self.dt)
        reward = goal_seeking_reward(self.dynamics, self.goal, action, self.dt)
        self.tick += 1
        done = self.tick > self.ep_len
        if self.scene is not None:
            self.scene.chase_cam.step(self.dynamics.pos, self.dynamics.vel)
        sv = self.dynamics.state_vector()
        return sv, reward, done, {}

    def _reset(self):
        self.goal = npa(0, 0, 2)
        x, y = self.np_random.uniform(-self.box, self.box, size=(2,))
        x = -abs(x)
        y = 0
        if self.box < 20:
            self.box *= 1.0003 # x20 after 10000 resets
        z = self.np_random.uniform(1, 3)
        pos = npa(x, y, z)
        vel = omega = npa(0, 0, 0)
        #rotz = np.random.uniform(-np.pi, np.pi)
        #rotation = r3d.rotz(rotz)
        #rotation = rotation[:3,:3]
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.tick = 0
        return self.dynamics.state_vector()

    def _render(self, mode='human', close=False):
        if self.scene is None:
            self.scene = Quadrotor3DScene(self.goal, self.dynamics,
                640, 480, resizable=True)
        self.scene.update_state(self.dynamics)
        return self.scene.render_firstperson(return_rgb_array=(mode == 'rgb_array'))


class QuadrotorVisionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        self.controller = ShiftedMotorControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene3p = None
        self.scene1p = None

        seq_len = 4
        img_w, img_h = 64, 64
        img_space = spaces.Box(-1, 1, (img_h, img_w, seq_len))
        imu_space = spaces.Box(-100, 100, (6, seq_len))
        self.observation_space = spaces.Tuple([img_space, imu_space])
        self.img_buf = np.zeros((img_w, img_h, seq_len))
        self.imu_buf = np.zeros((6, seq_len))

        # TODO get this from a wrapper
        self.ep_len = 10000
        self.tick = 0
        self.dt = 1.0 / 50.0

        self._seed()

        # size of the box from which initial position will be randomly sampled
        # grows a little with each episode
        self.box = 1.0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.controller.step(self.dynamics, action, self.dt)
        reward = goal_seeking_reward(self.dynamics, self.goal, action, self.dt)
        self.tick += 1
        done = self.tick > self.ep_len
        if self.scene3p is not None:
            self.scene3p.chase_cam.step(self.dynamics.pos, self.dynamics.vel)

        self.scene1p.update_state(self.dynamics)
        rgb = self.scene1p.render_firstperson(return_rgb_array=True)
        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        self.img_buf = np.roll(self.img_buf, -1, axis=2)
        self.img_buf[:,:,-1] = grey

        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        self.imu_buf = np.roll(self.imu_buf, -1, axis=1)
        self.imu_buf[:,-1] = imu

        return (self.img_buf, self.imu_buf), reward, done, {}

    def _reset(self):
        self.goal = npa(0, 0, 2)
        x, y = self.np_random.uniform(-self.box, self.box, size=(2,))
        x = -abs(x)
        y = 0
        if self.box < 20:
            self.box *= 1.0003 # x20 after 10000 resets
        z = self.np_random.uniform(1, 3)
        pos = npa(x, y, z)
        vel = omega = npa(0, 0, 0)
        #rotz = np.random.uniform(-np.pi, np.pi)
        #rotation = r3d.rotz(rotz)
        #rotation = rotation[:3,:3]
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)
        if self.scene1p is None:
            w, h, _ = self.img_buf.shape
            self.scene1p = Quadrotor3DScene(self.goal, self.dynamics,
                w, h, resizable=False, visible=False) #TODO deal with retina display
        self.scene1p.update_state(self.dynamics)

        # fill the buffers with copies of initial state
        w, h, seq_len = self.img_buf.shape
        rgb = self.scene1p.render_firstperson(return_rgb_array=True)
        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        self.img_buf = np.tile(grey[:,:,None], (1,1,seq_len))
        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        self.imu_buf = np.tile(imu[:,None], (1,seq_len))

        self.tick = 0
        return (self.img_buf, self.imu_buf)

    def _render(self, mode='human', close=False):
        if self.scene3p is None:
            self.scene3p = Quadrotor3DScene(self.goal, self.dynamics,
                640, 480, resizable=True)
        self.scene3p.update_state(self.dynamics)
        return self.scene3p.render_chase(return_rgb_array=(mode == 'rgb_array'))
        return None
