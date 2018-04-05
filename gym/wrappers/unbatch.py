from gym import Wrapper
import numpy as np

class UnBatcher(Wrapper):
	def _step(self, a):
		obs, rewards, dones, info = self.env.step(a[None,:])
		assert info is None # TODO
		return obs[0,:], rewards[0], dones[0], None

	def _reset(self):
		obs = self.env.reset()
		return obs[0,:]

	def sample_sysid(self):
		self.env.sample_sysid()

	def sysid_values(self):
		sysid = self.env.sysid_values()
		return sysid[0,:]

class BatchCycler(Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.i = 0
		self.N = env.N

	def _step(self, a):
		acs = np.tile(a, (self.N, 1))
		obs, rewards, dones, info = self.env.step(acs)
		assert info is None # TODO
		assert np.all(dones == dones[0])
		return obs[0,:], rewards[0], dones[0], None

	def _reset(self):
		self.env.sample_sysid()
		obs = self.env.reset()
		return obs[0,:]

	def sample_sysid(self):
		self.env.sample_sysid()

	#def sysid_values(self):
		#sysid = self.env.sysid_values()
		#return sysid[i,:]
