import random
import numpy as np 

class NaiveMech(object):
	"""
	This is the naive mechanism that simply posts a fixed price for all arriving
	data.
	"""
	def __init__(self, alg, T=0, B=0, price_threshold=1.0):
		self.alg = alg
		self.reset(0.1, T, B, price_threshold=price_threshold)
		super(NaiveMech, self).__init__()

	# we don't need to know cmax, but for compatibility with the interface
	def reset(self, eta, T, B, cmax=1.0, price_threshold=1.0):
		self.T = T
		self.B = B
		self.price_threshold = price_threshold
		self.spend = 0.0
		self.alg.reset(eta)

	def train_and_get_err(self, costs, Xtrain, Ytrain, Xtest, Ytest):
		# train
		for i in xrange(len(Xtrain)):
			if costs[i] <= self.price_threshold:
				self.spend += self.price_threshold
				self.alg.data_update(Xtrain[i], Ytrain[i], 1.0)
				if self.spend >= self.B:
					break
			else:
				self.alg.null_update()
		# get err
		return self.alg.test_error(Xtest, Ytest)

