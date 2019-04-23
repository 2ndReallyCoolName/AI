from Optimizations.Losses import Losses
import numpy as np
from nn.rnn.RNNCell import RNNCell


class DynamicRNN:
    def __init__(self, cell):
        self.Cell = cell
        self.L = Losses()

    def dynamic_feed_forward(self, sequence):
        sequence = np.asarray(sequence)
        assert sequence.ndim == 2
        hidden_vect = np.zeros(shape=self.Cell.hidden_size)
        for i in range(len(sequence)):
            self.Cell.Vectors['h'+str(i)] = hidden_vect
            self.Cell.timestamp = i+1
            self.Cell.feed_forward(sequence[i])
            hidden_vect = self.Cell.Vectors['h'+str(i+1)]
        return self.Cell.Vectors['c'+str(self.Cell.timestamp)]

    # def back_prop(self, loss_vector, seq_start, seq_end, training_rate):

    def fit(self, x, y, loss_function='quadratic', training_rate=0.5, back_prop_length=0, alpha=1):
        assert x.ndim == 2
        c = self.dynamic_feed_forward(x)
        seq_end = self.Cell.timestamp
        if back_prop_length == 0:
            back_prop_length = seq_end-1
        layer_error = np.ones(self.Cell.I.input_shape)
        seq_start = seq_end - back_prop_length
        total_cost = 0
        while seq_start > 0:
            loss_vector, cost = self.L.loss_function(loss_function, y, c, alpha)
            np.multiply(layer_error, self.Cell.back_prop(loss_vector, seq_start=seq_start, seq_end=seq_end,
                                                         training_rate=training_rate))
            c = self.dynamic_feed_forward(x)
            seq_end, seq_start = seq_end - back_prop_length, seq_start - back_prop_length
            total_cost += cost

        print(total_cost)
        return layer_error


if __name__ == '__main__':
    D = DynamicRNN(RNNCell(10, 10, 4))
    seq = []
    for i in range(10):
        seq.append(np.random.random(size=10))

    print(D.dynamic_feed_forward(seq))

