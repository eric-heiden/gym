import numpy as np
import gym
from copy import deepcopy
from gym import utils
from gym.envs.mujoco import ReacherEnv
from gym.envs.mujoco.reacher_xml import ReacherXML

N_RAND = 1000

def expand_obs_by(obs, n):
    infs = np.full(n, np.inf)
    low = np.concatenate([obs.low, -infs])
    high = np.concatenate([obs.high, infs])
    return gym.spaces.Box(low, high)

class ReacherBatchEnv(gym.Env):

    def __init__(self, N=1):
        self.N = N
        self.sysid_dim = 4
        self.tick = 0
        self._seed()

        # construct a bunch of randomized models
        envs_all = []
        sysid_vecs = []
        xr = ReacherXML()
        for i in range(N_RAND):
            xr.randomize(self.np_random)
            env = ReacherEnv(model_path=xr.get_path(),
                min_radius=xr.min_rad, max_radius=xr.max_rad)
            env.original_index = i
            envs_all.append(env)
            sysid_vecs.append(xr.sysid_values()[None,:])

        self.envs_all = np.array(envs_all)
        self.sysid_all = np.concatenate(sysid_vecs, 0)
        print("sysid_vecs[0].shape:", sysid_vecs[0].shape)
        print("sysid_all.shape:", self.sysid_all.shape)
        assert self.sysid_all.shape == (N_RAND, self.sysid_dim)

        env0 = self.envs_all[0]
        self.action_space = env0.action_space
        self.observation_space = expand_obs_by(
            env0.observation_space, self.sysid_dim)
        self.envs = None
        self.sample_sysid()

        # rendering stuff
        self.metadata = deepcopy(env0.metadata)

    def _seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def sample_sysid(self):
        prev_env0 = None
        if self.envs is not None:
            prev_env0 = self.envs[0]
        selection = self.np_random.choice(N_RAND, self.N)
        self.envs = self.envs_all[selection]
        self.sysid = self.sysid_all[selection,:]
        assert self.sysid.shape == (self.N, self.sysid_dim)
        if prev_env0 is not None and prev_env0.viewer is not None:
            self.envs[0]._take_viewer(prev_env0)

    def sysid_values(self):
        return deepcopy(self.sysid)

    def _step(self, a):
        #print("acs:", a)
        #a[a > 1] = 1
        #a[a < -1] = -1
        obs = np.zeros((self.N, self.envs[0].obs_dim))
        rewards = np.zeros(self.N)
        # TODO merge reward dicts
        for i in range(self.N):
            ob, reward, done, rew_dict = self.envs[i]._step(a[i,:])
            obs[i,:] = ob
            rewards[i] = reward

        self.tick += 1
        done = self.tick >= 50
        dones = np.full(self.N, done)
        if done:
            self.tick = 0
            for env in self.envs:
                env.reset()

        obs = np.concatenate([obs, self.sysid], axis=1)

        return obs, rewards, dones, None

    def _render(self, mode='human', close=False):
        self.envs[0].render(mode=mode, close=close)

    def _reset(self):
        for env in self.envs:
            env.reset()
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.N, self.envs[0].obs_dim))
        for i in range(self.N):
            obs[i,:] = self.envs[i]._get_obs()
        return np.concatenate([obs, self.sysid], 1)
