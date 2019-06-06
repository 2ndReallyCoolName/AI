import numpy as np


class Window:
    def __init__(self, window_dimensions=(3, 3)):
        self.win_cols, self.win_rows = window_dimensions

    def get_window(self, array2d, start_row, start_col):
        assert array2d.ndim == 2
        window = np.zeros((self.win_rows * self.win_cols))
        for row in range(self.win_rows):
            r = row + start_row
            if r >= 0:
                for col in range(self.win_cols):
                    if col+start_col >= 0:
                        try:
                            window[(row*self.win_cols)+col] = array2d[r, col+start_col]
                        except IndexError:
                            pass
        return window


if __name__ == '__main__':
    # I = Image.open(r'C:\Users\ninanpyo\Pictures\19983784_1247165272076953_2765720285955350871_o.jpg')
    z = [i for i in range(16)]
    z = np.reshape(z, (4,4))
    c = Window((3, 3))
    w = c.get_window(z, 1, 2)
    print(z)
    print(w)
    pass
