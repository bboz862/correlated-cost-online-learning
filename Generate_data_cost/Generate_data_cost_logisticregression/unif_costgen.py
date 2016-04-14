import random

class UnifCostGen(object):
	"""
	Generates costs to go with the data.
	The marginal distribution is always uniform [0,1].
	(You can scale all returned values by cmax if desired.)
  Initialized with a list of which labels are expensive and which are cheap
  (all others are independent uniform [0,1])
	Let a = (num_cheap) / (num_cheap + num_exp).
	Cheap data is priced uniform [0, a]
	Expensive data is priced uniform [a, 1]
	Thus the marginal distribution is uniform [0,1].
	"""
	def __init__(self, seed, expensive=[], cheap=[]):
		self.randgen = random.Random(seed)
		self.label_types = [0]*10  # 1 -> expensive, -1 -> cheap
		for l in expensive:
			self.label_types[l] = 1
		for l in cheap:
			self.label_types[l] = -1
		self.frac_cheap = 0.5  # don't know until normalize() is called
		super(UnifCostGen, self).__init__()

	# re-adjust cost of hi and lo points so that marginal is uniform
	# num_examples[l] = number of l-labels in the dataset
	def normalize(self, num_examples):
		num_cheap = sum([1 if self.label_types[i] == -1 else 0 for i,n in enumerate(num_examples)])
		num_exp = sum([1 if self.label_types[i] == 1 else 0 for i,n in enumerate(num_examples)])
		if num_cheap + num_exp == 0:
			self.frac_cheap = 0.5  # vacuously true
		else:
			self.frac_cheap = float(num_cheap)/float(num_exp + num_cheap)

	def draw_cost(self, label):
		unif_rand = self.randgen.random()
		if self.label_types[label] == 1:
			return self.frac_cheap + unif_rand * (1.0 - self.frac_cheap)
		elif self.label_types[label] == -1:
			return unif_rand * self.frac_cheap
		else:
			return unif_rand

