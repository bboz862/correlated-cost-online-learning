import random
import numpy as np 
from scipy.optimize import brentq

# This is a generalization of RothSchoenebeck12, where we apply our
# importance-weighting framework, but we get the probabilities of
# purchase by drawing prices as prescribed by RothSchoenebeck12.

# RS12 assumes knowledge of the prior distribution on costs and
# tailors the price distribution accordingly.
# Here we assume the cost prior is uniform [0,cmax].

# Computing the pricing distribution:
# With a cost CDF F, PDF f, the pricing distribution of RS12 has the form
# Pr[ price >= x ] = sqrt(alpha f(x) / (F(x) + xf(x)))   (or 1 if larger than 1).
# (in general there is smoothing etc, but this isn't needed for the uniform cost case.)
# The uniform[0,cmax] distribution has F(x) = x/cmax and f(x) = 1/cmax.
# Pr[ price >= x ] = sqrt(alpha 1/(2x)).
#                  = beta sqrt(1/x)  by a change of variables.
# So we always set a price in the range [beta^2, cmax] with a point mass at cmax.
# The pricing pdf is g(x) = beta / (2x^1.5).
# To solve for beta, the expected spend per arrival must be B/T.
# So B/T  =  Pr[price=cmax]*cmax  +  int_{beta^2}^{cmax} x g(x) F(x) dx,
# which works out to beta*sqrt(cmax)  +  (1/3)(beta*sqrt(cmax) - beta^4/cmax).
#    = (4/3)*beta*sqrt(cmax) - (1/3)*beta^4/cmax
# So we just solve for the beta where this equals B/T.


class RS12Mech(object):
	"""
	This is the RothSchoenebeck2012 mechanism that draws prices i.i.d. from a fixed
	  distribution for all arriving data.
	Generalized to interact with a learning algorithm by giving its importance
	  weight.
	Assumes the marginal distribution on costs is uniform [0,cmax].
	"""
	def __init__(self, alg, seed, T=1, B=1, eta=0.1, cmax=1.0):
		self.alg = alg
		self.randgen = random.Random(seed)
		self.reset(eta, T, B, cmax)
		super(RS12Mech, self).__init__()

	def reset(self, eta, T, B, cmax=1.0):
		self.T = T
		self.B = B
		self.cmax = cmax
		if cmax <= B/float(T):  # can buy every point
			self.factor = 100000000000
		else:
			self.factor = brentq(lambda x: 4.0*x*(cmax**0.5)/3.0 - (x**4.0)/(3.0*cmax) - B/float(T), 0.0, cmax**0.5)
		self.spend = 0.0
		self.alg.reset(eta)

	# given quantile ~ unif[0,1], return a price such that
	#   Pr[price >= c] = factor/sqrt(c)
	def _get_price(self, quantile):
		val = self.factor / quantile
		return min(self.cmax, val ** 2.0)

	def _prob_exceeds(self, c):
		if c == 0.0:
			return 1.0
		return min(1.0, self.factor / c**0.5)

	def train_and_get_err(self, costs, Xtrain, Ytrain, Xtest, Ytest):
		for i in xrange(len(Xtrain)):
			price = self._get_price(self.randgen.random())
			if costs[i] <= price:
				self.spend += price
				self.alg.data_update(Xtrain[i], Ytrain[i], self._prob_exceeds(costs[i]))
# could cut it off at B, but it's designed to spend B in expectation so don't
#				if self.spend >= self.B:
#					break
			else:
				self.alg.null_update()
		return self.alg.test_error(Xtest, Ytest)

