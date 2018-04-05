import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, model_path='reacher.xml', min_radius=0.0, max_radius=0.2):
        self.min_radius = min_radius
        self.max_radius = max_radius
        # penalize control effort less when links are heavier
        # TODO: deal with this by changing actuator strength to match links
        # in mj model, and normalize control range to [-1, 1]
        self.ctrl_penalty_scale = (0.2 / max_radius) ** 2
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, model_path, 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_vel = - 0.0010 * np.square(self.model.data.qvel.flat[:2]).sum()
        reward_ctrl = - self.ctrl_penalty_scale * np.square(a).sum()
        reward = reward_dist + reward_vel + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        # don't want very tight initial angle on elbow
        joint_angles = self.np_random.uniform(low=[-np.pi, -2.0], high=[np.pi, 2.0]) + self.init_qpos[:2]

        # goal slides along y-axis - i.e. fixing coord system rotation wrt goal
        minr = max(self.min_radius, 0.05)
        gy = self.np_random.uniform(low=minr, high=self.max_radius)
        self.goal = np.array([0, gy])

        qpos = np.concatenate([joint_angles, self.goal])
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
