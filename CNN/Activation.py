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
