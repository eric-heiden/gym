import numpy as np
import matplotlib.pyplot as plt
import sys, imageio
from transforms3d.quaternions import quat2mat, qinverse

from gym.envs.classic_control.quadrotor_modular import QuadrotorEnv
from gym.envs.classic_control import rendering3d as r3d

from tqdm import tqdm

import pyglet.gl
from pyglet.gl import *


class LineStrip:
    def __init__(self):
        self.vertices = []

    def add(self, vertex):
        self.vertices.append(vertex)

    def draw(self):
        glDisable(GL_LIGHTING)
        glBegin(GL_LINE_STRIP)
        glColor3f(1, 1, 1)
        glLineWidth(3.)
        for vertex in self.vertices:
            glVertex3f(vertex[0], vertex[1], vertex[2])
        glEnd()
        glEnable(GL_LIGHTING)


def plot_trajectory(traj_filename: str):
    fig = plt.figure()
    xyz = np.array([tuple(float(coord) for coord in line.split()[1:4]) for line in open(traj_filename, 'r')])

    # 1 unit is about 100 meters
    xyz *= 10.
    # swap y and z, flip the new z and move up
    xyz[:,1], xyz[:,2] = xyz[:,2], -xyz[:,1].copy()+20

    # quaternions are in the format qx, qz, qy, qw
    # convert to transforms3d.quaternions, i.e. qw, qx, qy, qz
    quats = np.array([tuple(float(coord) for coord in line.split()[4:]) for line in open(traj_filename, 'r')])
    assert quats.shape[1] == 4

    permutation = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]])
    rotations = [np.dot(r3d.rotz(-np.pi/4)[:3,:3], np.dot(quat2mat(quats[i, :]), permutation)) for i in range(quats.shape[0])]
    # for i in range(quats.shape[0]):
    #     quats[i, :] = qinverse(quats[i, :])
    # quats[:, 1], quats[:, 2] = quats[:, 1], -quats[:, 2].copy()
    # quats[:, 0], quats[:, 3] = quats[:, 3], quats[:, 0].copy()

    env = QuadrotorEnv()
    while True:
        env.reset()
        lines = LineStrip()
        env.scene.scene.batches.append(lines)
        for t in tqdm(range(xyz.shape[0]-1)):
            env.dynamics.set_state(position=xyz[t,:],
                                   velocity=xyz[t+1,:]-xyz[t,:],
                                   rotation=rotations[t],
                                   omega=np.zeros(3))
            lines.add(xyz[t,:])
            env.step(np.zeros(env.action_space.shape))
            env.render()


if __name__ == '__main__':
    traj_filename = '/home/eric/.deep-rl-docker/james_gym/KeyFrameTrajectory.txt' if len(sys.argv) < 2 else sys.argv[1]
    plot_trajectory(traj_filename)
