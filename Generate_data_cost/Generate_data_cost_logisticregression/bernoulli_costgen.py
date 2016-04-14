import random

class BernoulliCostGen(object):
	"""
	Generates costs to go with the data.
	Initialized with a list of which labels are are exp (expensive)
	and a probability p to output a exp label.
	Marginal cost distribution is 0 with probability 1-p
    and 1 with probability p.	Makes as many 'exp' labels expensive as
    possible subject to this marginal distribution.
	"""
	def __init__(self, seed, p = 0.1, expensive=[]):
		self.randgen = random.Random(seed)
		self.label_types = [0]*10  # 1 -> expensive, -1 -> cheap
		for l in expensive:
			self.label_types[l] = 1
		self.p = p
		self._set_params(0.5)
		super(BernoulliCostGen, self).__init__()

	# p_exp: probability, on a exp point, to output cost 1
	# p_nonexp: ditto for nonexp point
	def _set_params(self, frac_exp):
		if frac_exp == 0.0:
			self.p_exp = self.p  # doesn't matter since there are no exp points
			self.p_nonexp = self.p
		elif frac_exp >= self.p:
			# in this case, a nonexpensive point never gets a cost of 1
      # and an expensive point may sometimes have cost 0
			self.p_nonexp = 0.0
			# with probability frac_exp we have an exp data point and then with
			# probability p_exp we give it cost 1
			self.p_exp = self.p / frac_exp
		else:
			self.p_exp = 1.0
			self.p_nonexp = (self.p - frac_exp) / (1.0 - frac_exp)

	def normalize(self, num_examples):
		num_exp = sum([1 if self.label_types[i] == 1 else 0 for i,n in enumerate(num_examples)])
		frac_exp = float(num_exp)/float(sum(num_examples))
		self._set_params(frac_exp)

	def draw_cost(self, label):
		if self.label_types[label] == 1:  # expensive type
			threshold = self.p_exp
		else:
			threshold = self.p_nonexp
		if self.randgen.random() <= threshold:
			return 1.0
		else:
			return 0.0

