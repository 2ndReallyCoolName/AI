from nn.Dense import Dense
import numpy as np
from nn.Activation import Activation


class RNNCell:
    """Vectors: all hidden and cell state vectors"""
    Vectors = {}
    timestamp = 1

    def __init__(self, input_size, hidden_size, training_iterations=5, hidden_activation='softmax', weight_param=(-1, 1),
                 bias_params=(0, 1), bias_bool=True, fp=''):
        self.hidden_size = hidden_size
        self.hidden_activation = hidden_activation
        self.I = Dense(input_size, hidden_size, None, weight_param, bias_params, False)
        self.H = Dense(hidden_size, hidden_size, None, weight_param, bias_params, bias_bool)
        self.iterations = training_iterations
        self.A = Activation()
        if fp == '':
            self.initiate_weights()

        self.Vectors['h0'] = np.zeros(hidden_size)
        self.dh_dW1 = np.zeros(shape=(input_size, hidden_size))
        self.dh_dWh = np.zeros(shape=(hidden_size, hidden_size))
        self.dh_db = np.zeros(hidden_size)

    def initiate_weights(self):
        self.I.initialize(), self.H.initialize()

    def feed_forward(self, input_vect):
        h1 = self.I.feed_forward(input_vect)
        h2 = self.H.feed_forward(self.Vectors['h'+str(self.timestamp-1)])
        h = np.add(h1, h2)
        y = self.A.activation_function(h, self.hidden_activation, "a"+str(self.timestamp))

        self.Vectors['x' + str(self.timestamp)] = input_vect
        self.Vectors['h'+str(self.timestamp)] = h
        self.Vectors['y'+str(self.timestamp)] = y
        self.timestamp += 1

    '''returns none'''
    def gradient(self, error_vector, timestamp, training_rate=1):
        assert timestamp < self.timestamp
        wh = self.H.Weight

        ev = np.multiply(error_vector, training_rate)
        max_timestamp = min(timestamp+self.iterations, self.timestamp)

        while timestamp < max_timestamp:
            x = self.Vectors['x' + str(timestamp)]
            h_prv = self.Vectors['h' + str(timestamp - 1)]

            for j in range(self.hidden_size):
                error = ev[j]
                activation_error = self.A.errors["a" + str(timestamp)][j]
                self.H.Bias[j] += error*self.dB1(activation_error, wh[j][j], self.dh_db[j], j)
                for i in range(self.I.input_shape):
                    self.I.Weight[i][j] += error*self.dW1(activation_error, x[i], wh[j][j], self.dh_dW1[i][j], i, j)
                for j2 in range(self.hidden_size):
                    self.H.Weight[j2][j] += error * self.dWh(activation_error, h_prv[j2], wh[j2][j], self.dh_dWh[j2][j],
                                                            j2, j)
            timestamp += 1

    def dW1(self, ae, x, wh, dh_prv, i, j):
        er = (x + wh*dh_prv)*ae
        self.dh_dW1[i][j] = er
        return er

    def dWh(self, ae, h_prv, wh, dh_prv, i, j):
        er = ae*(h_prv + wh*dh_prv)
        self.dh_dWh[i][j] = er
        return er

    def dB1(self, ae, wh, dh_prv, j):
        er = ae*(wh*dh_prv + 1)
        self.dh_db[j] = er
        return er

    def reset(self):
        self.timestamp = 1
        self.dh_dW1 = np.zeros(shape=(self.I.input_shape, self.hidden_size))
        self.dh_dWh = np.zeros(shape=(self.hidden_size, self.hidden_size))
        self.dh_db = np.zeros(self.hidden_size)

    def get_output(self):
        return self.Vectors['y'+str(self.timestamp-1)]


if __name__ == '__main__':
    s, t = 0, 100
    for _ in range(t):
        i, o = 20, 5
        R = RNNCell(i, o)
        vects = [np.random.random(size=i) for j in range(4)]

        y = np.zeros(o)
        y[1] = 1

        for vect in vects:
            R.feed_forward(vect)

        for i in range(20):
            error_vect = np.subtract(y, R.get_output())
            R.gradient(error_vect, 1)
            R.reset()

            for vect in vects:
                R.feed_forward(vect)

        if np.argmax(R.get_output()) == 1:
            s += 1
    print(s/t)