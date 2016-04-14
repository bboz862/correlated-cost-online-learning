import random
import numpy as np 

class GradientDescent(object):
    """
    Implementation of a gradient descent algorithm for binary classification.
    The loss function is 1 if classified incorrectly, 0 if correctly,
      but use a hinge loss for training.
    Initialized with a list of digits to be labeled positively
      (all others negatively).
    The algorithm initializes self.w to be a vector of 0's.
    Then for each of a sequence of rounds, check if the loss is positive;
        if so do a gradient descent update.
    Supports methods:
        reset(eta)
        norm_grad_loss(x, y)
        test_error(X, Y)
        data_update(x, y, importance_wt)
        null_update()
    """
    def __init__(self, num_features, pos_labels, eta=0.1):
        # num_features = length of the x vector
        # pos_labels   = list of digits in [0,9] to label positively (others negatively)
        self.num_features = num_features
        self.binarized_labels = [1.0 if l in pos_labels else -1.0 for l in range(10)]
        self.reset(eta)
        super(GradientDescent, self).__init__()

    def reset(self, eta):
        self.eta = eta
        self.w = np.zeros(self.num_features)
        self.avg_w = self.w
        self.steps = 0
    
    # predict on only a single datapoint, using avg_w
    def _predict_one(self,x):
        return np.sign(self.avg_w.dot(x))

    # predict on a matrix of data (shape is numdata * numfeatures)
    def _predict(self,X):
        Ypreds = np.apply_along_axis(self._predict_one, 1, X)
        return Ypreds

    # predict and report error on data (shape is numdata * numfeatures)
    def test_error(self, X, Y):
        Ypred = self._predict(X)
        return np.mean(map(lambda y : self.binarized_labels[y], Y) != Ypred)

    def _loss(self,x,biny):
        score = self.w.dot(x) * biny
        return max(1. - score, 0.)

    def _grad_loss(self, x, biny):
        # gradient of the loss
        loss = self._loss(x,biny)
        if loss > 0.:
            return -biny * x
        return np.zeros(self.num_features)

    def norm_grad_loss(self, x, y):
        # l2 norm of the gradient of the loss
        return np.linalg.norm(self._grad_loss(x,self.binarized_labels[y]))

    def _step(self):
        # every step, whether we got a data_update or a null_update, we need to
        # step the average
        self.avg_w = (self.avg_w * self.steps + self.w) / float(self.steps + 1.0)
        self.steps += 1

    # Do an importance-weighted gradient descent update on the hypothesis
    def data_update(self, x, y, importance_wt):
        biny = self.binarized_labels[y]
        loss = self._loss(x,biny)
        if loss > 0.:
            step = self.eta * self._grad_loss(x, biny)
            self.w = self.w - step / importance_wt
        self._step()

    # Do an update with no new data
    def null_update(self):
        self._step()

