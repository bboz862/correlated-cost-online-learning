import random
import numpy as np 

class BaselineMech(object):
	"""
	This is the baseline mechanism that takes every point without having to pay for it.
	"""
	def __init__(self, alg):
		self.alg = alg
		self.reset(0.1, 1, 1)
		super(BaselineMech, self).__init__()

	# we don't need to know T, B, cmax: it's just for interface compatibility
	def reset(self, eta, T, B, cmax=1.0):
		self.alg.reset(eta)

	def train_and_get_err(self, costs, Xtrain, Ytrain, Xtest, Ytest):
		for i in xrange(len(Xtrain)):
			self.alg.data_update(Xtrain[i], Ytrain[i], 1.0)
		return self.alg.test_error(Xtest, Ytest)

