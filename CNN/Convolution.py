import numpy as np
from nn.cnn.Filters import Filters
from nn.cnn.window import Window


class ConvActivations:
    def __init__(self, alpha=0.2, func='leaky_relu'):
        self.alpha = alpha
        self.func = func

    @staticmethod
    def relu(number):
        return max(0, number)

    def leaky_relu(self, number):
        return self.alpha*number if number < 0 else number

    @staticmethod
    def d_relu(number):
        return 0 if number < 0 else 1

    def d_leaky_relu(self, number):
        return self.alpha if number < 0 else 1

    def activation_function(self, number):
        dct = {'relu': self.relu, 'leaky_relu': self.leaky_relu}
        return dct[self.func](number)

    def activation_function_derivative(self, number):
        dct = {'relu': self.d_relu, 'leaky_relu': self.d_leaky_relu}
        return dct[self.func](number)


class Convolution2D:
    current_layers = []
    # shape : (filters, conv_height, conv_width, total window size)
    gradients = []

    def __init__(self, prev_layers, image_dimensions, stride=2, window_dimensions=(3, 3), pool_size=4,
                 filter_params=(0, 2), bias_params=(0, 1), number_filters=3, new_filters=True, bias=True,
                 activation='relu', alpha=0.2):
        self.prev_layers = prev_layers
        self.img_width, self.img_height = image_dimensions
        self.win_height, self.win_width = window_dimensions
        self.Filter = Filters(window_dimensions, filter_params, bias_params, number_filters, new_filters, bias)
        self.Window = Window(window_dimensions)
        self.Activation = ConvActivations(alpha=alpha, func=activation)
        self.convolute(stride)
        self.__get_current_dimensions__()
        self.MaxPool = MaxPooling(self.conv_width, self.conv_height, pool_size)
        self.__max_pool__()

    def __wrapper__(self, layer, stride, f):
        crr_layer, gradient_layer = [], []
        row, col = 0, 0
        while row < self.img_height:
            curr_row, gradient_row = [], []
            while col < self.img_width:
                window = self.Window.get_window(layer, row, col)
                fltr = self.Filter.Filters['f'+f]
                rst = np.dot(window, fltr) + self.Filter.Filters['b'+f]
                curr_row.append(rst)
                gradient_row.append(self.Activation.activation_function_derivative(rst)*window)
                col += stride
            row += stride
            col = 0
            crr_layer.append(np.asarray(curr_row))
            gradient_layer.append(np.asarray(gradient_row))
        return np.asarray(crr_layer), np.asarray(gradient_layer)

    def convolute(self, stride):
        for f in range(self.Filter.num):
            for layer in self.prev_layers:
                layers = self.__wrapper__(layer, stride, str(f))
                self.current_layers.append(layers[0])
                self.gradients.append(layers[1])

    def __max_pool__(self):
        new_layers = []
        for layer in self.current_layers:
            new_layers.append(self.MaxPool.pool(layer))
        self.pooled_width, self.pooled_height = self.MaxPool.pool_width, self.MaxPool.pool_height
        self.current_layers = new_layers
        self.pooled_gradients = self.MaxPool.gradients
        self.pooled_size = self.MaxPool.pool_size

    def __get_current_dimensions__(self):
        layer = self.current_layers[0]
        self.conv_width= len(layer[0])
        self.conv_height = len(layer)

    def __gradients__(self, loss, l, m, fltr, j, training_rate):
        gradient = loss*self.gradients[fltr][l][m][j]
        loss *= self.Filter.Filters['f'+fltr][j]
        self.Filter.Filters['f'+fltr][j] += gradient*training_rate
        return loss

    def gradient_handler(self, loss, training_rate=0.5):
        fltr_limit, filter_size = len(self.gradients), self.win_width*self.win_height
        print(self.pooled_height)
        for n in range(self.pooled_height):
            for o in range(self.pooled_width):
                dM = self.pooled_gradients[n][o]
                loss = dM*loss
                for _pr in range(self.pooled_size):
                    l, m = n, o
                    for _pc in range(self.pooled_size):
                        for fltr in range(fltr_limit):
                            for i in range(filter_size):
                                yield self.__gradients__(loss, l, m, fltr, i, training_rate)
                                return 1
                        m += 1
                    l += 1


class MaxPooling:
    gradients = []

    def __init__(self, width, height, pool_size=4):
        self.pool_size = pool_size
        self.W = Window((pool_size, pool_size))
        self.inp_width, self.inp_height = width, height
        self.__set_ouput_dimensions__(width, height)

    def __set_ouput_dimensions__(self, width, height):
        self.pool_width, self.pool_height = int(width / self.pool_size), int(height / self.pool_size)

    def pool(self, layer):
        print(layer)
        gradient = np.zeros((self.inp_height, self.inp_width))
        new_layer = np.zeros((self.pool_height, self.pool_width))
        row = 0
        for _row in range(self.pool_height):
            col = 0
            for _col in range(self.pool_width):
                window = self.W.get_window(layer, row, col)
                num = window.max()
                pos = window.argmax()
                new_layer[_row, _col] = num
                gradient[row + int((pos / self.pool_size)), col + (pos % self.pool_size)] = 1
                col += self.pool_size
            row += self.pool_size
        self.gradients.append(gradient)
        print('pool')
        print(new_layer)
        return new_layer


class Flatten:
    def __init__(self, layers):
        self.vector = self.flatten(layers)

    @staticmethod
    def flatten(layers):
        vect = []
        for layer in layers:
            for row in layer:
                vect.extend(row)
        return np.asarray(vect)


class Convolution3D(Convolution2D):
    def __init__(self, pixel_array, image_dimensions, stride=2, window_dimensions=(3, 3), pool_size=4,
                 filter_params=(-1, 1), bias_params=(-1, 1), number_filters=3, new_filters=True, bias=True,
                 activation='relu', alpha=0.2):
        layer = self.__reformat__(pixel_array, image_dimensions)
        super().__init__(layer, image_dimensions, stride, window_dimensions, pool_size, filter_params, bias_params,
                         number_filters, new_filters, bias, activation, alpha)

    @staticmethod
    def __reformat__(array, size):
        array = np.asarray(array)
        l, dims = [], 1
        try:
            dims = len(array[0])
        except TypeError:
            pass
        for i in range(dims):
            l.append(array[:, i].reshape(size))
        return l


if __name__ == '__main__':
    w, h = 2, 2
    l1 = np.reshape([i for i in range(0, h*w)], (h, w))
    C = Convolution2D([l1], (w, h), 1, window_dimensions=(2, 2))
    # print('_________________________ FILTER _______________________________')
    # print(C.Filter.Filters)
    # print('_________________________ CONV _______________________________')
    # print(C.current_layers)
    # print('')
    # print('_________________________ CONV GRADIENTS_______________________________')
    # # gradient = C.gradients
    # # for g in gradient:
    # #     print(g)
    # # # print(C.gradients)
    # layer = [i for i in range(25)]
    # layer = np.reshape(layer, (5, 5))
    # M = MaxPooling(5, 5, 2)
    # # # print(M.pool(C.relu_layers[0], C.crr_width, C.crr_height))
    # m1 = M.pool(layer)
    # print(layer)
    # print(m1)
    # print(M.gradients)
    # # print('_________________________ M1 _______________________________')
    # # m2 = M.pool(C.relu_layers[1], C.crr_width, C.crr_height)
    # # print(m2)
    # # print('_________________________ M2 _______________________________')
    # # F = Flatten([m1,m2])
    # # print(F.flatten([m1,m2]))
    # # print('_________________________ FLATTEN _______________________________')
    g = list(C.gradient_handler(1))
    print(g)
    # for error in C.gradient_handler(1):
    #     print(error)
