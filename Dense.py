from nn.Variables import Variables
from nn.Activation import Activation
from Optimizations.Losses import Losses
import numpy as np


class Dense(Variables):

    def __init__(self, input_size, unit_size, activation='softmax', weight_params=(1, 5), bias_params=(0, 1),
                 bias_bool=True, store_vectors=True):
        self.A = Activation()
        self.Vectors = {}
        self.store_vectors = store_vectors
        super().__init__(unit_size, input_size, activation, weight_params, bias_params, bias_bool)
        if not bias_bool:
            self.bias_params = (0, 0)

    def feed_forward(self, input_vect):
        x = np.asarray(input_vect)
        h = np.add(np.matmul(x, self.Weight), self.Bias)
        y = self.A.activation_function(h, self.activation)
        if self.store_vectors:
            self.Vectors['x'], self.Vectors['h'], self.Vectors['y'] = x, h, y
        return y

    def gradient(self, error, i, j, training_rate=1, momentum_bool=0, gamma=0.9, last_layer=True):
        assert isinstance(error, float) or isinstance(error, int)
        dy_dh = self.A.errors[j]
        dh_w = self.Vectors['x'][i]
        dh_dx = self.Weight[i][j]
        cost_w, cost_b = error * dy_dh * training_rate, error * dy_dh * training_rate
        if not last_layer:
            cost_w *= dh_w
        delta = momentum_bool*self.W_p_mtx[i][j] + cost_w
        self.Weight[i][j] += delta
        self.errors[i][j] += delta
        self.Bias[j] += momentum_bool*self.B_p_mtx[j] + cost_b
        if momentum_bool:
            self.momentum_handler(i, j, cost_w, cost_b, gamma)
        return error*dy_dh*dh_dx

    def momentum_handler(self, i, j, cost_w, cost_b, gamma):
        self.W_p_mtx[i, j] = gamma*self.W_p_mtx[i, j] + cost_w
        self.B_p_mtx[j] = gamma*self.B_p_mtx[j] + cost_b
        return


vocab_size = 5
encoded_vect_size = 5
epochs = 5


data_size = 100


def word2vectGenerator():
    lst = np.zeros(vocab_size)
    lst[np.random.randint(0, vocab_size)] = 1
    return lst


def labelGenerator():
    indices = np.random.randint(0, vocab_size, np.random.randint(1, vocab_size-1))
    lst = np.zeros(vocab_size)
    for index in indices:
        lst[index] = 1
    return lst


if __name__ == "__main__":
    D = Dense(2, 4)
    print(D.feed_forward([1, 2]))



    # x = word2vectGenerator()
    # y = labelGenerator()

    # L = Losses()
    # last_vect = []
    # for times in range(10):
    #     predicted = D.feed_forward(x)
    #     loss_vector = L.loss_function("quadratic", y, predicted)
    #     print(predicted)
    #     # print(y)
    #     # print(loss_vector[0])
    #     last_vect = predicted
    #     # print("_________________________________")
    #     for i in range(len(y)):
    #         for j in range(len(x)):
    #             D.gradient(loss_vector[0][i], j, i)
    #
    # print(y)
    # # print(np.argmax(last_vect))
    # # print(D.errors)

