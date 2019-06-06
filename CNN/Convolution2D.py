from nn.cnn.Filters import Filters
from nn.cnn.Activation import ConvActivations
from nn.cnn.window import Window
import numpy as np
import time


class Conv2D:
    def __init__(self, stride=2, window_dimensions=(3, 3),
                 filter_params=(0, 2), bias_params=(0, 1), number_filters=3, new_filters=True, bias=True,
                 activation='relu', alpha=0.2, padding=0):
        self.win_height, self.win_width = window_dimensions
        self.padding = padding
        self.stride = stride
        self.Filter = Filters(window_dimensions, filter_params, bias_params, number_filters, new_filters, bias)
        self.Window = Window(window_dimensions)
        self.Activation = ConvActivations(alpha=alpha, func=activation)
        self.conv_width, self.conv_height = (0, 0)

    def convolute(self, image):
        image = np.asarray(image)
        h, w = image.shape
        h += self.padding
        w += self.padding
        row = -self.padding
        layers = [[] for _ in range(self.Filter.num)]
        while row < h:
            rws = [[] for _ in range(self.Filter.num)]
            col = -self.padding
            while col < w:
                win = self.Window.get_window(image, row, col)
                for f in range(self.Filter.num):
                    rws[f].append(np.add(np.dot(win, self.Filter.Filters["f"+str(f)]), self.Filter.Filters["b"+str(f)]))
                col += self.stride

            for i in range(self.Filter.num):
                layers[i].append(rws[i])
            row += self.stride
        self.conv_width, self.conv_height = np.shape(layers[0])
        return layers

    def print(self, arr):
        for row in arr:
            print(row)
        print("<-------------->")


if __name__ == "__main__":
    C = Conv2D(padding=0, window_dimensions=(3, 3), stride=10)
    im = np.random.random((5000, 5000))
    t = time.time()
    c1 = C.convolute(im)
    print(time.time() - t)

