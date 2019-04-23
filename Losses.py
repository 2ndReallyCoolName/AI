import numpy as np


class Losses(object):
    losses = None
    t, gamma = 1, 0

    def quadratic_cost(self, predicted, expected):
        self.quadratic_cost_gradient(predicted, expected)
        return 0.5*np.sum(np.square(predicted-expected))

    def cross_entropy(self, predicted, expected):
        assert isinstance(predicted, np.ndarray) and isinstance(expected, np.ndarray)
        self.cross_entropy_gradient(predicted, expected)
        return - np.sum(np.multiply(expected, np.log(predicted)) + (1 - expected)*np.log(1 - predicted))

    def exponential_cost(self, predicted, expected):
        return self.t*np.exp((1/self.t)*np.sum(predicted, expected))

    def exponential_cost_wrapper(self, predicted, expected):
        self.exponential_cost_gradient(predicted, expected)
        return self.exponential_cost(predicted, expected)

    def quadratic_cost_gradient(self, predicted, expected):
        self.losses = expected - predicted

    def cross_entropy_gradient(self, predicted, expected):
        self.losses = (predicted - expected)/(np.multiply((1 - predicted), predicted))

    def exponential_cost_gradient(self, predicted, expected):
        self.losses = (2/self.t)*np.multiply((predicted - expected), self.exponential_cost(predicted, expected))

    def loss_function(self, loss_name, expected, predicted, *args):
        dct = {'quadratic':self.quadratic_cost, 'cross_entropy': self.cross_entropy,
               'exponential': self.exponential_cost}
        try:
            self.t = args[0]
            self.gamma = args[1]
        except IndexError:
            pass
        cost = np.sum(dct[loss_name](predicted, expected))
        return self.losses, cost

