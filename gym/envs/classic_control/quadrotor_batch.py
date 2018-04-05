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
from gym.envs.classic_control import rendering3d as r3d
import numpy as np
from numpy.linalg import norm
import traceback
import sys
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
    d = m.shape[-1]
    prod = np.matmul(m, np.swapaxes(m, -1, -2))
    # broadcasting eye across last 2 dimensions
    return np.max(np.abs(prod - np.eye(d)).flatten()) < 0.00001

def normalize(x):
    n = norm(x)
    if n < 0.00001:
        return x, 0
    return x / n, n

# gram-schmidt orthogonalization
def orth_cols(m):
    if len(m.shape) == 2:
        m = deepcopy(m)
        _, nc = m.shape
        m[:,0] = m[:,0] / norm(m[:,0])
        for i in range(1, nc):
            c = m[:,i]
            left = m[:,:i]
            proj = np.matmul(left.T, c)
            c = c - np.matmul(left, proj)
            clen = norm(c)
            assert clen > 0.90
            m[:,i] = c / clen
        return m
    else:
        N, _, _ = m.shape
        morth = 0 * m
        for i in range(N):
            morth[i,:,:] = orth_cols(m[i,:,:])
        return morth

# gram-schmidt orthogonalization - trying to do it faster w vectorized operations,
# but not working yet@!!
def orth_cols_vectorized_wrong(m):
    raised_dim = False
    if len(m.shape) == 2:
        raised_dim = True
        m = m[None,:,:]

    _, _, nc = m.shape
    m[:,:,0] = m[:,:,0] / norm(m[:,:,0], axis=0)
    for i in range(1, nc):
        print("orth_cols", i)
        col = m[:,:,i]
        print("col shape:", col.shape)
        rest = m[:,:,:i]
        print("rest shape:", rest.shape)
        rT = rest.transpose([0,2,1])
        cB = col[:,:,None]
        proj = np.matmul(rT, cB)
        print("proj shape:", proj.shape)
        subtr = np.matmul(rest, proj)
        print("subtr shape:", subtr.shape)
        col = col - np.matmul(rest, proj).squeeze()
        print("col shape:", col.shape)
        clen = norm(col, axis=1)
        print("clen shape:", clen.shape)
        # TODO: this should really be a tighter threshold,
        # matrices from dRdt should only be slightly perturbed from orthonormal
        assert np.all(clen > 0.7), "error at {}".format(np.where(clen <= 0.7))
        m[:,:,i] = col / clen[:,None]
    if raised_dim:
        m = m.squeeze()
    assert is_orthonormal(m)
    return m

def rand_uniform_rot3d(np_random, N=1):
    if N == 1:
        randunit = lambda: normalize(np_random.normal(size=(3,)))[0]
        up = randunit()
        fwd = randunit()
        while np.dot(fwd, up) > 0.95:
            fwd = randunit()
        left = normalize(np.cross(up, fwd))
        up = np.cross(fwd, left)
        rot = np.hstack([fwd, left, up])
    else:
        rot = np.zeroes((N,3,3))
        for i in range(N):
            rot[i,:,:] = rand_uniform_rot3d(np_random, 1)
    return rot

def npa(*args):
    return np.array(args)

def omega_mat_deriv(omega):
    if omega.size == 3:
        x, y, z = omega
        dRdt = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    else:
        # TODO can make more elegant?
        N, d = omega.shape
        assert d == 3
        x, y, z = [a.squeeze() for a in np.hsplit(omega, 3)]
        dRdt = np.zeros((N, 3, 3))
        dRdt[:,0,1] = -z
        dRdt[:,0,2] = y

        dRdt[:,1,0] = z
        dRdt[:,1,2] = -x

        dRdt[:,2,0] = -y
        dRdt[:,2,1] = -x
    return dRdt


def hinge_loss(x, loss_above):
    h = x - loss_above
    h[h < 0] = 0
    return h

class QuadrotorDynamics(object):
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    def __init__(self, mass, arm_length, inertia, thrust_to_weight, torque_to_thrust=0.05):
        N = mass.size
        for param in (arm_length, thrust_to_weight):
            assert param.shape == (N,)
        assert inertia.shape == (N, 3)

        self.N = N
        self.mass = mass
        self.arm = arm_length
        self.inertia = inertia
        self.thrust_constant = GRAV * mass * thrust_to_weight / 4.0
        self.torque_constant = torque_to_thrust * self.thrust_constant
        scl = arm_length / norm([1,1,0])
        self.prop_pos = np.outer(scl, np.array([
            [1,  1, -1, -1],
            [1, -1, -1,  1],
            [0,  0,  0,  0]]).T.flatten()).reshape((N, 4, 3))
        self.prop_ccw = np.array([1, -1, 1, -1])

    # pos, vel, in world coords
    # rotation is (body coords) -> (world coords)
    # omega in body coords
    def set_state(self, position, velocity, rotation, omega, thrusts=None):
        N = self.N
        if thrusts is None:
            thrusts = np.zeros((N, 4))
        for v in (position, velocity, omega):
            assert v.shape == (N,3)
        assert thrusts.shape == (N,4)
        assert rotation.shape == (N,3,3)
        assert is_orthonormal(rotation)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega)
        self.thrusts = deepcopy(thrusts)
        self.crashed = np.full(N, False)

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, np_random, box, vel_max=15.0, omega_max=2*np.pi):
        N = self.N
        pos = np_random.uniform(low=-box, high=box, size=(N,3))
        vel = np_random.uniform(low=-vel_max, high=vel_max, size=(N,3))
        omega = np_random.uniform(low=-omega_max, high=omega_max, size=(N,3))
        rot = rand_uniform_rot3d(np_random, N)
        self.set_state(pos, vel, rot, omega)

    def step(self, thrust_cmds, dt):

        crash = self.pos[:,2] <= self.arm
        self.pos[crash,2] = self.arm[crash]
        self.vel[crash,:] = 0
        self.omega[crash,:] = 0
        self.crashed = crash

        N = self.N
        assert thrust_cmds.shape == (N,4)
        assert np.all(thrust_cmds.flatten() >= 0)
        assert np.all(thrust_cmds.flatten() <= 1)
        per_motor_thrust = self.thrust_constant[:,None] * thrust_cmds
        thrust_vec = np.outer(np.sum(thrust_cmds, axis=1), [0, 0, 1])
        torque = per_motor_thrust[:,:,None] * np.cross(self.prop_pos, [0, 0, 1])
        torque[:,:,2] += self.torque_constant[:,None] * (self.prop_ccw * thrust_cmds)
        #torques = np.cross(self.prop_pos, npa(0,0,1)) * thrusts[:,None]
        #torques[:,2] += self.torque_constant * self.prop_ccw * thrust_cmds
        torque = np.sum(torque, axis=1)
        assert torque.shape == (N, 3)

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
        thrust_world = np.matmul(self.rot, thrust_vec[:,:,None]).squeeze()
        acc = [0, 0, -GRAV] + (1.0 / self.mass[:,None]) * thrust_world 
        self.vel = vel_damp * self.vel + dt * acc
        self.pos = self.pos + dt * self.vel

    def state_vector(self):
        rot_flat = self.rot.reshape(self.N, 9)
        # add phony sysid obseration
        return np.hstack([self.pos, self.vel, rot_flat, self.omega, np.zeros((self.N,2))])

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


class QuadrotorBatchEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        N = 32
        self.sysid_dim = 2
        self.N = N
        np.seterr(all='raise')
        N1 = np.ones(N)
        mass = 0.5 * N1
        arm_length = (0.33 / 2.0) * N1
        inertia = np.outer(mass, [0.01, 0.01, 0.02])
        self.thrust_to_weight = 2.0 * N1
        self.dynamics = QuadrotorDynamics(mass, arm_length, inertia,
            thrust_to_weight=self.thrust_to_weight)

        self.ep_len = 256
        self.tick = 0
        self.dt = 1.0 / 100.0

        self.action_space = spaces.Box(np.zeros(4), np.ones(4))
        # pos, vel, rot, omega
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim + self.sysid_dim) # one for dummy sysid
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        self._seed()
        self.reset()
        self.viewer = None
        self.model = None

    def sample_sysid(self):
        pass

    def sysid_values(self):
        return np.zeros((self.N, self.sysid_dim))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        N = self.N
        box = 10
        xy = self.np_random.uniform(-box, box, size=(N, 2))
        #x, y = 20, 0
        z = self.np_random.uniform(1, 3, size=(N, 1))
        pos = np.hstack([xy, z])
        vel = np.zeros((N, 3))
        omega = np.zeros((N, 3))
        rotz = np.random.uniform(-np.pi, np.pi, size=N)
        rotation = np.zeros((N, 3, 3))
        for i in range(N):
            rotation[i,:,:] = r3d.rotz(rotz[i])[:3,:3]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.camera = ChaseCamera(pos=pos[0,:], vel=vel[0,:])
        self.goal = np.tile([0, 0, 2], (N, 1))
        self.tick = 0
        np.set_printoptions(precision=4)
        return self.dynamics.state_vector()

    def _step(self, action):
        N = self.N
        assert action.shape == (N, 4)
        # noop for gfx testing
        #return self.dynamics.state_vector(), 0, False, {}
        #action = (action + 1.0) / self.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        #action[:] = 0.5
        #action = npa(0.6, 0.4, 0.6, 0.4)
        self.dynamics.step(action, self.dt)
        self.camera.step(self.dynamics.pos[0,:], self.dynamics.vel[0,:])

        self.tick += 1
        done = np.full(self.N, self.tick >= self.ep_len)

        dist = norm(self.goal - self.dynamics.pos, axis=1)
        loss_pos = dist
        loss_alt = 2 * hinge_loss(self.dynamics.pos[:,2], 3) ** 2
        loss_spin = 0.2 * np.sum(np.abs(self.dynamics.omega), axis=1)
        loss_crash = 50 * self.dynamics.crashed
        loss_battery = 0.2*np.sum(action, axis=1)

        goal_thresh = 4.0 # within this distance, start rewarding
        goal_max = 50 # max reward when exactly at goal
        a = -goal_max / (goal_thresh**2)
        reward_goal = a * dist**2 + goal_max
        reward_goal[reward_goal < 0] = 0

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
            loss_pos, loss_alt, loss_spin, loss_crash, loss_battery], axis=0)

        sv = self.dynamics.state_vector()
        return sv, reward, done, {}

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
            diameter = 2 * self.dynamics.arm[0]

            self.quad_transform = quadrotor_3dmodel(diameter)

            self.shadow_transform = r3d.transform_and_color(
                np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

            floor = r3d.CheckerTexture(
                r3d.rect((1000, 1000), (0, 100), (0, 100)))

            goal = r3d.transform_and_color(r3d.translate(self.goal[0,:]),
                (0.5, 0.4, 0), r3d.sphere(diameter/2, 18))

            #world = r3d.World([
               #self.quad_transform, self.shadow_transform, floor, goal])
            world = r3d.BackToFront([
                floor, self.shadow_transform, goal, self.quad_transform])
            batch = r3d.Batch()
            world.build(batch, None)

            self.viewer.add_batch(batch)

        pos, rot = self.dynamics.pos[0,:], self.dynamics.rot[0,:,:]
        matrix = r3d.trans_and_rot(pos, rot)
        self.quad_transform.set_transform(matrix)

        shadow_pos = 0 + pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform(matrix)

        eye, center, up = self.camera.look_at()
        self.viewer.look_at(eye, center, up)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')
