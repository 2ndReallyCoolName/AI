import numpy as np


class Window:
    def __init__(self, window_dimensions=(3, 3)):
        self.win_cols, self.win_rows = window_dimensions

    def get_window(self, array, start_row, start_col):
        assert array.ndim == 2
        window = np.zeros((self.win_rows * self.win_cols))
        for row in range(self.win_rows):
            for col in range(self.win_cols):
                try:
                    window[(row*self.win_cols)+col] = array[row+start_row, col+start_col]
                except IndexError:
                    break
        return window


if __name__ == '__main__':
    # I = Image.open(r'C:\Users\ninanpyo\Pictures\19983784_1247165272076953_2765720285955350871_o.jpg')
    z = [i for i in range(16)]
    z = np.reshape(z, (4,4))
    c = Window((2, 2))
    w = c.get_window(z, 2, 3)
    print(z)
    print(w)
    pass
