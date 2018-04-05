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
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

GRAV = 9.81
TILES = 256 # number of tiles used for the obstacle map

# overall TODO:
# - fix front face CCW to enable culling
# - add texture coords to primitives
# - add more controllers
# - oracle policy (access to map, true state, etc.)
# - non-flat floor
# - fog

# numpy's cross is really slow for some reason
def cross(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

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
    left = normalize(cross(up, fwd))
    up = cross(fwd, left)
    rot = np.hstack([fwd, left, up])
    return rot

def npa(*args):
    return np.array(args)

def hinge_loss(x, loss_above):
    try:
        return np.max(0, x - loss_above)
    except TypeError:
        return max(0, x - loss_above)

def clamp_norm(x, maxnorm):
    n = np.linalg.norm(x)
    return x if n <= maxnorm else (maxnorm / n) * x


class QuadrotorDynamics(object):
    # thrust_to_weight is the total, it will be divided among the 4 props
    # torque_to_thrust is ratio of torque produced by prop to thrust
    def __init__(self, mass, arm_length, inertia, thrust_to_weight=2.0, torque_to_thrust=0.05):
        assert np.isscalar(mass)
        assert np.isscalar(arm_length)
        assert inertia.shape == (3,)
        # unit: kilogram
        self.mass = mass
        # unit: meter
        self.arm = arm_length
        # unit: kg * m^2
        self.inertia = inertia
        # unit: ratio
        self.thrust_to_weight = thrust_to_weight
        self.thrust = GRAV * mass * thrust_to_weight / 4.0
        self.torque = torque_to_thrust * self.thrust
        scl = arm_length / norm([1,1,0])
        self.prop_pos = scl * np.array([
            [1,  1, -1, -1],
            [1, -1, -1,  1],
            [0,  0,  0,  0]]).T # row-wise easier with np
        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0, 0, 1])
        # 1 for props turning CCW, -1 for CW
        self.prop_ccw = np.array([1, -1, 1, -1])
        self.since_last_svd = 0

    # pos, vel, in world coords
    # rotation is (body coords) -> (world coords)
    # omega in body coords
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3,3)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega)
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, np_random, box, vel_max=15.0, omega_max=2*np.pi):
        pos = np_random.uniform(low=-box, high=box, size=(3,))
        vel = np_random.uniform(low=-vel_max, high=vel_max, size=(3,))
        omega = np_random.uniform(low=-omega_max, high=omega_max, size=(3,))
        rot = rand_uniform_rot3d(np_random)
        self.set_state(pos, vel, rot, omega)

    def step(self, thrust_cmds, dt):
        # uncomment for debugging. they are slow
        #assert np.all(thrust_cmds >= 0)
        #assert np.all(thrust_cmds <= 1)
        thrusts = self.thrust * thrust_cmds
        torques = self.prop_crossproducts * thrusts[:,None]
        torques[:,2] += self.torque * self.prop_ccw * thrust_cmds
        torque = np.sum(torques, axis=0)
        thrust = npa(0,0,np.sum(thrusts))

        # TODO add noise

        vel_damp = 0.999
        omega_damp = 0.999

        # rotational dynamics
        omega_dot = ((1.0 / self.inertia) *
            (cross(-self.omega, self.inertia * self.omega) + torque))
        self.omega = omega_damp * self.omega + dt * omega_dot

        x, y, z = self.omega
        omega_mat_deriv = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

        dRdt = np.matmul(omega_mat_deriv, self.rot)
        self.rot += dt * dRdt
        self.since_last_svd += 1
        if self.since_last_svd > 60:
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
        up = cross(to, R[:,1])
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

# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust * dynamics.prop_crossproducts.T
    torque[2,:] = dynamics.torque * dynamics.prop_ccw
    thrust = dynamics.thrust * np.ones((1,4))
    dw = (1.0 / dynamics.inertia)[:,None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    assert np.linalg.cond(J) < 25.0
    return J

# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low  = npa(min_g, -max_rp, -max_rp, -max_yaw)
        high = npa(max_g,  max_rp,  max_rp,  max_yaw)
        return spaces.Box(low, high)

    def step(self, dynamics, action, dt):
        kp = 5.0 # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0 # meters / sec
        dymax = 4 * np.pi # radians / sec
        high = npa(vmax, vmax, vmax, dymax)
        return spaces.Box(-high, high)

    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:,0]))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return npa(R[2,1], R[0,2], R[1,0])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = npa(0, 0, action[3])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, dynamics.rot[:,2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


class NonlinearPositionController2(object):
    def __init__(self, dynamics):
        self.vel = VelocityYawControl(dynamics)

    def step(self, dynamics, goal, dt):
        kv = 1.2
        v_des = -kv * clamp_norm(dynamics.pos - goal, 4.0)

        x, y, _ = dynamics.rot[:,0]
        theta = np.arctan2(y, x)
        ky = 0.0
        vy_des = -ky * theta

        action = np.append(v_des, vy_des)
        self.vel.step(dynamics, action, dt)


class NonlinearPositionController(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def step(self, dynamics, goal, dt):
        kp_p, kd_p = 5.0, 4.0
        kp_a, kd_a = 100.0, 50.0

        e_p = clamp_norm(dynamics.pos - goal, 4.0)
        e_v = dynamics.vel
        acc_des = -kp_p * e_p - kd_p * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, [1, 0, 0]))
        xb_des    = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return npa(R[2,1], R[0,2], R[1,0])
        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_w = dynamics.omega

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_des, dynamics.rot[:,2])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO:
# class AttitudeControl
# class VelocityControl

def bulky_goal_seeking_reward(dynamics, goal, action, dt):
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

    loss_effort = 0.02 * norm2(action)

    # TODO this is too hacky, try not to use it
    goal_thresh = 1.0 # within this distance, start rewarding
    goal_max = 0 # max reward when exactly at goal
    a = -goal_max / (goal_thresh**2)
    reward_goal = max(0,  a * norm2(to_goal) + goal_max)

    reward = -dt * np.sum([
        -reward_goal,
        loss_pos, loss_vel_away, loss_alt, loss_spin, loss_effort])

    return reward

def goal_seeking_reward(dynamics, goal, action, dt):
    # log to create a sharp peak at the goal
    dist = np.linalg.norm(goal - dynamics.pos)
    loss_pos = np.log(dist + 0.1) + 0.1 * dist

    # penalize altitude above this threshold
    max_alt = 6.0
    loss_alt = np.exp(2*(dynamics.pos[2] - max_alt))

    # penalize amount of control effort
    loss_effort = 0.0 * np.linalg.norm(action)

    reward = -dt * np.sum([loss_pos, loss_alt, loss_effort])
    return reward


class ChaseCamera(object):
    def __init__(self):
        self.view_dist = 4

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.999
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel

        veln, n = normalize(self.vel_smooth)
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        if False:  # or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
            # look towards goal even though we are not heading there
            right, _ = normalize(cross(ideal_vel, up))
        else:
            right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * back + 0.3 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (to_eye + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up


# determine where to put the obstacles such that no two obstacles intersect
# and compute the list of obstacles to collision check at each 2d tile.
def _place_obstacles(np_random, N, box, radius_range, our_radius, tries=5):

    t = np.linspace(0, box, TILES+1)[:-1]
    scale = box / float(TILES)
    x, y = np.meshgrid(t, t)
    pts = np.zeros((N, 2))
    dist = x + np.inf

    radii = np_random.uniform(*radius_range, size=N)
    radii = np.sort(radii)[::-1]
    test_list = [[] for i in range(TILES**2)]

    for i in range(N):
        rad = radii[i]
        ok = np.where(dist.flat > rad)[0]
        if len(ok) == 0:
            if tries == 1:
                print("Warning: only able to place {}/{} obstacles. "
                    "Increase box, decrease radius, or decrease N.")
                return pts[:i,:], radii[:i]
            else:
                return place_obstacles(N, box, radius_range, tries-1)
        pt = np.unravel_index(np_random.choice(ok), dist.shape)
        pt = scale * np.array(pt)
        d = np.sqrt((x - pt[1])**2 + (y - pt[0])**2) - rad
        # big slop factor for tile size, off-by-one errors, etc
        for ind1d in np.where(d.flat <= 2*our_radius + scale)[0]:
            test_list[ind1d].append(i)
        dist = np.minimum(dist, d)
        pts[i,:] = pt - box/2.0

    # very coarse to allow for binning bugs
    test_list = np.array(test_list).reshape((TILES, TILES))
    #amt_free = sum(len(a) == 0 for a in test_list.flat) / float(test_list.size)
    #print(amt_free * 100, "pct free space")
    return pts, radii, test_list

# generate N obstacles w/ randomized primitive, size, color, TODO texture
# arena: boundaries of world in xy plane
# our_radius: quadrotor's radius
def _random_obstacles(np_random, N, arena, our_radius):
    arena = float(arena)
    # all primitives should be tightly bound by unit circle in xy plane
    boxside = np.sqrt(2)
    box = r3d.box(boxside, boxside, boxside)
    sphere = r3d.sphere(radius=1.0, facets=16)
    cylinder = r3d.cylinder(radius=1.0, height=2.0, sections=32)
    # TODO cone-sphere collision
    #cone = r3d.cone(radius=0.5, height=1.0, sections=32)
    primitives = [box, sphere, cylinder]

    bodies = []
    max_radius = 2.0
    positions, radii, test_list = _place_obstacles(
        np_random, N, arena, (0.5, max_radius), our_radius)
    for i in range(N):
        primitive = np_random.choice(primitives)
        tex_type = r3d.random_textype()
        tex_dark = 0.5 * np_random.uniform()
        tex_light = 0.5 * np_random.uniform() + 0.5
        color = 0.5 * np_random.uniform(size=3)
        heightscl = np.random.uniform(0.5, 2.0)
        height = heightscl * 2.0 * radii[i]
        z = (0 if primitive is cylinder else
            (height/2.0 if primitive is sphere else
            (height*boxside/4.0 if primitive is box
            else np.nan)))
        translation = np.append(positions[i,:], z)
        matrix = np.matmul(r3d.translate(translation), r3d.scale(radii[i]))
        matrix = np.matmul(matrix, np.diag([1, 1, heightscl, 1]))
        body = r3d.Transform(matrix,
            #r3d.ProceduralTexture(tex_type, (tex_dark, tex_light), primitive))
                r3d.Color(color, primitive))
        bodies.append(body)

    return ObstacleMap(arena, bodies, test_list)


class ObstacleMap(object):
    def __init__(self, box, bodies, test_lists):
        self.box = box
        self.bodies = bodies
        self.test = test_lists

    def detect_collision(self, dynamics):
        pos = dynamics.pos
        if pos[2] <= dynamics.arm:
            print("collided with terrain")
            return True
        r, c = self.coord2tile(*dynamics.pos[:2])
        if r < 0 or c < 0 or r >= TILES or c >= TILES:
            print("collided with wall")
            return True
        if self.test is not None:
            radius = dynamics.arm + 0.1
            return any(self.bodies[k].collide_sphere(pos, radius)
                for k in self.test[r,c])
        return False

    def sample_start(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((pad, pad + band), np_random)

    def sample_goal(self, np_random):
        pad = 4
        band = TILES // 8
        return self.sample_freespace((-(pad + band), -pad), np_random)

    def sample_freespace(self, rowrange, np_random):
        rfree, cfree = np.where(np.vectorize(lambda t: len(t) == 0)(
            self.test[rowrange[0]:rowrange[1],:]))
        choice = np_random.choice(len(rfree))
        r, c = rfree[choice], cfree[choice]
        r += rowrange[0]
        x, y = self.tile2coord(r, c)
        z = np_random.uniform(1.0, 3.0)
        return np.array([x, y, z])

    def tile2coord(self, r, c):
        #TODO consider moving origin to corner of world
        scale = self.box / float(TILES)
        return scale * np.array([r,c]) - self.box / 2.0

    def coord2tile(self, x, y):
        scale = float(TILES) / self.box
        return np.int32(scale * (np.array([x,y]) + self.box / 2.0))


class Quadrotor3DScene(object):
    def __init__(self, np_random, quad_arm, w, h,
        obstacles=True, visible=True, resizable=True):

        self.window_target = r3d.WindowTarget(w, h, resizable=resizable)
        self.obs_target = r3d.FBOTarget(64, 64)
        self.cam1p = r3d.Camera(fov=90.0)
        self.cam3p = r3d.Camera(fov=45.0)

        self.chase_cam = ChaseCamera()
        self.world_box = 40.0

        diameter = 2 * quad_arm
        self.quad_transform = self._quadrotor_3dmodel(diameter)

        self.shadow_transform = r3d.transform_and_color(
            np.eye(4), (0, 0, 0, 0.4), r3d.circle(0.75*diameter, 32))

        # TODO make floor size or walls to indicate world_box
        floor = r3d.ProceduralTexture(r3d.TEX_XOR, (0.15, 0.25),
            r3d.rect((1000, 1000), (0, 100), (0, 100)))

        self.goal_transform = r3d.transform_and_color(np.eye(4),
            (0.85, 0.55, 0), r3d.sphere(diameter/2, 18))

        self.map = None
        bodies = [r3d.BackToFront([floor, self.shadow_transform]),
            self.goal_transform, self.quad_transform]

        if obstacles:
            N = 20
            self.map = _random_obstacles(np_random, N, self.world_box, quad_arm)
            bodies += self.map.bodies

        world = r3d.World(bodies)
        batch = r3d.Batch()
        world.build(batch)

        self.scene = r3d.Scene(batches=[batch], bgcolor=(0,0,0))
        self.scene.initialize()

    def _quadrotor_3dmodel(self, diam):
        r = diam / 2
        prop_r = 0.3 * diam
        prop_h = prop_r / 15.0

        # "X" propeller configuration, start fwd left, go clockwise
        rr = r * np.sqrt(2)/2
        deltas = ((rr, rr, 0), (rr, -rr, 0), (-rr, -rr, 0), (-rr, rr, 0))
        colors = ((1,0,0), (1,0,0), (0,1,0), (0,1,0))
        def disc(translation, color):
            color = 0.5 * np.array(list(color)) + 0.2
            disc = r3d.transform_and_color(r3d.translate(translation), color,
                r3d.cylinder(prop_r, prop_h, 32))
            return disc
        props = [disc(d, c) for d, c in zip(deltas, colors)]

        arm_thicc = diam / 20.0
        arm_color = (0.6, 0.6, 0.6)
        arms = r3d.transform_and_color(
            np.matmul(r3d.translate((0, 0, -arm_thicc)), r3d.rotz(np.pi / 4)), arm_color,
            [r3d.box(diam/10, diam, arm_thicc), r3d.box(diam, diam/10, arm_thicc)])

        arrow_x = r3d.transform_and_color(r3d.rotx(-np.pi/2), (1,0,0), [r3d.arrow(0.12*prop_r, 2.5*prop_r, 16)])
        arrow_y = r3d.transform_and_color(r3d.roty(np.pi/2), (0,1,0), [r3d.arrow(0.12*prop_r, 2.5*prop_r, 16)])
        arrow_z = r3d.Color((0.2, 0.3, 0.9), r3d.arrow(0.12*prop_r, 2.5*prop_r, 16))

        bodies = props + [arms, arrow_x, arrow_y, arrow_z]
        self.have_state = False
        # return r3d.Transform(np.eye(4), bodies)
        return r3d.Transform(r3d.roty(-np.pi/2), bodies)

    # TODO allow resampling obstacles?
    def reset(self, goal, dynamics):
        self.goal_transform.set_transform(r3d.translate(goal))
        self.chase_cam.reset(goal, dynamics.pos, dynamics.vel)
        self.update_state(dynamics)

    def update_state(self, dynamics):
        self.have_state = True
        self.fpv_lookat = dynamics.look_at()
        self.chase_cam.step(dynamics.pos, dynamics.vel)

        matrix = r3d.trans_and_rot(dynamics.pos, dynamics.rot)
        self.quad_transform.set_transform_nocollide(matrix)

        shadow_pos = 0 + dynamics.pos
        shadow_pos[2] = 0.001 # avoid z-fighting
        matrix = r3d.translate(shadow_pos)
        self.shadow_transform.set_transform_nocollide(matrix)

        if self.map is not None:
            collided = self.map.detect_collision(dynamics)
        else:
            collided = dynamics.pos[2] <= dynamics.arm
        return collided

    def render_chase(self):
        assert self.have_state
        self.cam3p.look_at(*self.chase_cam.look_at())
        #self.cam3p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam3p, self.window_target)

    def render_obs(self):
        assert self.have_state
        self.cam1p.look_at(*self.fpv_lookat)
        r3d.draw(self.scene, self.cam1p, self.obs_target)
        return self.obs_target.read()


class QuadrotorEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        #self.controller = ShiftedMotorControl(self.dynamics)
        self.controller = OmegaThrustControl(self.dynamics)
        #self.controller = VelocityYawControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene = None

        self.oracle = NonlinearPositionController(self.dynamics)

        # pos, vel, rot, omega
        obs_dim = 3 + 3 + 9 + 3
        # TODO tighter bounds on some variables
        obs_high = 100 * np.ones(obs_dim)
        # rotation mtx guaranteed to be orthogonal
        obs_high[6:-3] = 1
        self.observation_space = spaces.Box(-obs_high, obs_high)

        # TODO get this from a wrapper
        self.ep_len = 64
        self.tick = 0
        self.dt = 1.0 / 50.0
        self.crashed = False

        self._seed()

        # size of the box from which initial position will be randomly sampled
        # grows a little with each episode
        self.box = 0.5

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.crashed:
            self.controller.step(self.dynamics, action, self.dt)
            #self.oracle.step(self.dynamics, self.goal, self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
            reward = goal_seeking_reward(self.dynamics, self.goal, action, self.dt)
        else:
            reward = -self.dt * 100
        self.tick += 1
        done = self.tick > self.ep_len# or self.crashed
        sv = self.dynamics.state_vector()
        return sv, reward, done, {}

    def _reset(self):
        if self.scene is None:
            self.scene = Quadrotor3DScene(None, self.dynamics.arm,
                640, 480, resizable=True, obstacles=False)

        self.goal = npa(0, 0, 2)
        x, y = self.np_random.uniform(-self.box, self.box, size=(2,))
        if self.box < 10:
            # from 0.5 to 10 after 100k episodes
            nextbox = self.box * 1.00003
            if int(4*nextbox) > int(4*self.box):
                print("box:", nextbox)
            self.box = nextbox
        #z = self.np_random.uniform(1, 3)
        z = self.goal[2]
        pos = npa(x, y, z)
        #pos = npa(0,0,2)
        vel = omega = npa(0, 0, 0)
        #vel = self.np_random.uniform(-2, 2, size=3)
        #vel[2] *= 0.1
        #rotz = np.random.uniform(-np.pi, np.pi)
        #rotation = r3d.rotz(rotz)
        #rotation = rotation[:3,:3]
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)

        self.scene.reset(self.goal, self.dynamics)
        self.scene.update_state(self.dynamics)

        self.crashed = False
        self.tick = 0
        if self.ep_len < 1000:
            self.ep_len += 0.01 # len 1000 after 100k episodes
        return self.dynamics.state_vector()

    def _render(self, mode='human', close=False):
        self.scene.render_chase()


class QuadrotorVisionEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        np.seterr(under='ignore')
        self.dynamics = default_dynamics()
        #self.controller = ShiftedMotorControl(self.dynamics)
        self.controller = OmegaThrustControl(self.dynamics)
        self.action_space = self.controller.action_space(self.dynamics)
        self.scene = None
        self.crashed = False

        seq_len = 4
        img_w, img_h = 64, 64
        img_space = spaces.Box(-1, 1, (img_h, img_w, seq_len))
        imu_space = spaces.Box(-100, 100, (6, seq_len))
        # vector from us to goal projected onto world plane and rotated into
        # our "looking forward" coordinates, and clamped to a maximal length
        dir_space = spaces.Box(-4, 4, (2, seq_len))
        self.observation_space = spaces.Tuple([img_space, imu_space, dir_space])
        self.img_buf = np.zeros((img_w, img_h, seq_len))
        self.imu_buf = np.zeros((6, seq_len))
        self.dir_buf = np.zeros((2, seq_len))

        # TODO get this from a wrapper
        self.ep_len = 500
        self.tick = 0
        self.dt = 1.0 / 50.0

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.crashed:
            self.controller.step(self.dynamics, action, self.dt)
            self.crashed = self.scene.update_state(self.dynamics)
            reward = goal_seeking_reward(self.dynamics, self.goal, action, self.dt)
        else:
            reward = -50
        self.tick += 1
        done = self.crashed or (self.tick > self.ep_len)

        rgb = self.scene.render_obs()
        # for debugging:
        #rgb = np.flip(rgb, axis=0)
        #plt.imshow(rgb)
        #plt.show()

        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        #self.img_buf = np.roll(self.img_buf, -1, axis=2)
        self.img_buf[:,:,:-1] = self.img_buf[:,:,1:]
        self.img_buf[:,:,-1] = grey

        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        #self.imu_buf = np.roll(self.imu_buf, -1, axis=1)
        self.imu_buf[:,:-1] = self.imu_buf[:,1:]
        self.imu_buf[:,-1] = imu

        # heading measurement - simplified, #95489c has a more nuanced version
        our_gps = self.dynamics.pos[:2]
        goal_gps = self.goal[:2]
        dir = clamp_norm(goal_gps - our_gps, 4.0)
        #self.dir_buf = np.roll(self.dir_buf, -1, axis=1)
        self.dir_buf[:,:-1] = self.dir_buf[:,1:]
        self.dir_buf[:,-1] = dir

        return (self.img_buf, self.imu_buf, self.dir_buf), reward, done, {}

    def _reset(self):
        if self.scene is None:
            self.scene = Quadrotor3DScene(self.np_random, self.dynamics.arm,
                640, 480, resizable=True)

        self.goal = self.scene.map.sample_goal(self.np_random)
        pos = self.scene.map.sample_start(self.np_random)
        vel = omega = npa(0, 0, 0)
        # for debugging collisions w/ no policy:
        vel = self.np_random.uniform(-20, 20, size=3)
        vel[2] = 0
        #rotz = np.random.uniform(-np.pi, np.pi)
        #rotation = r3d.rotz(rotz)
        #rotation = rotation[:3,:3]
        rotation = np.eye(3)
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.crashed = False

        self.scene.reset(self.goal, self.dynamics)
        collided = self.scene.update_state(self.dynamics)
        assert not collided

        # fill the buffers with copies of initial state
        w, h, seq_len = self.img_buf.shape
        rgb = self.scene.render_obs()
        grey = (2.0 / 255.0) * np.mean(rgb, axis=2) - 1.0
        self.img_buf = np.tile(grey[:,:,None], (1,1,seq_len))
        imu = np.concatenate([self.dynamics.omega, self.dynamics.accelerometer])
        self.imu_buf = np.tile(imu[:,None], (1,seq_len))

        self.tick = 0
        return (self.img_buf, self.imu_buf)

    def _render(self, mode='human', close=False):
        self.scene.render_chase()
