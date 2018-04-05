import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import sys, imageio
from transforms3d.quaternions import quat2mat, qinverse, rotate_vector

from tqdm import tqdm


def plot_trajectory(traj_filename: str):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    xyz = np.array([tuple(float(coord) for coord in line.split()[1:4]) for line in open(traj_filename, 'r')])
    ax.plot(xyz[:,0], xyz[:,2], -xyz[:,1])  # xyz[:,2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim([-3,0])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-2,1])

    quats = np.array([tuple(float(coord) for coord in line.split()[4:]) for line in open(traj_filename, 'r')])
    assert quats.shape[1] == 4

    # compute rotation matrices in correct coordinate system
    permutation = np.array([[0,  1, 0],
                            [-1,  0, 0],
                            [0, 0, 1]])
    rotations = [np.dot(quat2mat(quats[i, :]), permutation) for i in range(quats.shape[0])]

    # for i in range(quats.shape[0]):
    #     quats[i, :] = qinverse(quats[i, :])

    # quats[i, :] = qinverse(quats[i, :])
    # quats[:, 0], quats[:, 1] = -quats[:, 1], quats[:, 0].copy()
    # quats[:, 0] = -quats[:, 0]
    # quats[:, 1], quats[:, 2] = -quats[:, 2], quats[:, 1].copy()
    # quats[:, 3] = -quats[:, 3]
    # quats[:, 0], quats[:, 3] = quats[:, 3], quats[:, 0].copy()
    skip = 5
    arrow_length = 0.2

    # xyz arrows
    for base, color in zip([[1, 0, 0], [0, 1, 0], [0, 0, 1]], ['red', 'green', 'blue']):
        uvw = np.array([np.dot(rotation, base) for rotation in rotations])
        ax.quiver(xyz[::skip,0], xyz[::skip,2], -xyz[::skip,1],
                  uvw[::skip,0], uvw[::skip,1], uvw[::skip,2],
                  length=arrow_length, normalize=True, color=color)

    ax.scatter(xyz[0,0], xyz[0,2], -xyz[0,1], c='r', s=10)
    try:
        video = []
        for angle in tqdm(range(0, 360)):
            ax.view_init(30, angle)
            plt.draw()
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            img = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            video.append(img)
            # plt.pause(.001)
        imageio.mimsave('traj_visualization.mp4', video, fps=30)
    except:
        plt.show()


if __name__ == '__main__':
    traj_filename = 'KeyFrameTrajectory.txt' if len(sys.argv) < 2 else sys.argv[1]
    plot_trajectory(traj_filename)
