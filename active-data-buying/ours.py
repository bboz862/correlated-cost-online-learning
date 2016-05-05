import random
import numpy as np 

# Rather than initializing with prior knowledge, we simply estimate gamma
# online with a simple heuristic and set the normalizer K accordingly.

# Estimating gamma:
# We are using an importance-weighted estimate of gamma from the past data, i.e:
#   gamma = (1/(# steps))*(sum_{t < now} {0 if we did not observe t,  else this_round_gamma / q_t}
# where this_round_gamma = Delta(h_t,loss(.,z_t)) * (2sqrt(cmax) - sqrt(max(c*, c_t)))
# and   q_t = Pr[ we posted a price higher than c_t for z_t ].

# We also allow for discounting of past data by use of a discount factor.
# This allows for forgetting past data as things change. Thus, the actual estimate is
#   numerator = sum_{t <= s} discount^{s-t} {0 if not observe t, else .... }
#   denominator = sum_{t <= s} discount^{s-t}
#   gamma = numerator / denominator
# For the paper we used discount=1, which reduces to the first, simpler estimator.

# We initialize the numerator by 0 and denominator by a + b * T^c,
# which is intended to act as a regularizer so we start with some momentum to buy points
# and slowly decrease the buying rate as appropriate.


INIT_GAMMA_A = 10.0
INIT_GAMMA_B = 0.00001
INIT_GAMMA_C = 0.1
DISCOUNT = 1.0  # a discount factor for estimating gamma, if < 1, then old data is slowly forgotten

class OurMech(object):
    """
    This is our mechanism.
    It is initialized for either the usual setting (default),
  or "at-cost" (pay only the cost rather than your posted price).
    """
    def __init__(self, alg, seed, T=1, B=1, eta=0.1, cmax=1.0, atcost = False):
        self.alg = alg
        self.randgen = random.Random(seed)
        self.atcost = atcost
        self.reset(eta, T, B, cmax=cmax)
        super(OurMech, self).__init__()

    # cmax = maximum cost
    def reset(self, eta, T, B, cmax = 1.0):
        self.T = T
        self.B = B
        self.cmax = cmax
        self.spend = 0
        self.step = 0
        self.gamma_num = 0.0  # numerator
        self.gamma_den = INIT_GAMMA_A + INIT_GAMMA_B * float(T)**INIT_GAMMA_C
        self.gamma = self.gamma_num / self.gamma_den
        self.alg.reset(eta)

    # given quantile = Pr[price >= x], find and return x
    def _get_price(self, quantile, delta, K):
        if K == 0.0:
            return self.cmax
        val = delta / (K * quantile)
        return min(self.cmax , val ** 2.0)

    # return the probability price exceeds c
    def _prob_exceeds(self, c, delta, K):
        if K == 0.0 or c == 0.0:
            return 1.0
        return min(1.0 , delta / (K * c**0.5))

    # update our estimate of gamma (using it to set the normalizer K next round)
    def _update_gamma(self, cost, price, K, delta, prob_purchase):
        self.gamma_num *= DISCOUNT  # we will add something if we obtain the point
        self.gamma_den = self.gamma_den*DISCOUNT + 1.0
        if price >= cost:
            if self.atcost:
                self.gamma_num += delta * cost**0.5 / prob_purchase  # importance-weighted
            else:
                cstar = self.cmax if K == 0.0 else min(self.cmax, (delta / K)**2.0)
                self.gamma_num += delta * (2.0*self.cmax**0.5 - max(cost, cstar)**0.5) / prob_purchase
        self.gamma = self.gamma_num / self.gamma_den

    def _train(self, costs, Xtrain, Ytrain):
        for i in xrange(len(costs)):
            self.step += 1
            delta = self.alg.norm_grad_loss(Xtrain[i], Ytrain[i])
            c = costs[i]
            K = self.gamma * float(self.T - self.step) / (self.B - self.spend)
            quantile = self.randgen.random()
            price = self._get_price(quantile, delta, K)
            if price >= c:  # obtain the point
                prob_purchase = self._prob_exceeds(c, delta, K)
                self.alg.data_update(Xtrain[i], Ytrain[i], prob_purchase)
                if self.atcost:
                    self.spend += c
                else:
                    self.spend += price
                if self.spend >= self.B:
                    break
            else:
                self.alg.null_update()
            self._update_gamma(c, price, K, delta, prob_purchase)

    def train_and_get_err(self, costs, Xtrain, Ytrain, Xtest, Ytest):
        self._train(costs, Xtrain, Ytrain)
        return self.alg.test_error(Xtest, Ytest)


