import npm.utils as nu
import npm.proc as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from cycler import cycler
from scipy.optimize import curve_fit

# Base folder of the data for each curve
BASE = ['/local/home/fb250757/Documents/Data/Camera/cam2/2022_01_05/',
        '/local/home/fb250757/Documents/Data/Camera/cam2/2022_01_07/']

# filename for the curve 1
filenames_pos1 = ['cam2_001.h5', 'cam2_002.h5', 'cam2_003.h5',
                  'cam2_004.h5', 'cam2_005.h5', 'cam2_006.h5',
                  'cam2_007.h5', 'cam2_008.h5', 'cam2_009.h5',
                  'cam2_010.h5', 'cam2_011.h5', 'cam2_012.h5',
                  'cam2_013.h5', 'cam2_014.h5', 'cam2_015.h5',
                  'cam2_016.h5', 'cam2_017.h5', 'cam2_018.h5',
                  'cam2_019.h5', 'cam2_020.h5', 'cam2_021.h5',
                  'cam2_022.h5', 'cam2_023.h5', 'cam2_024.h5',
                  'cam2_025.h5', 'cam2_026.h5', 'cam2_027.h5',
                  'cam2_028.h5', 'cam2_029.h5', 'cam2_030.h5']

# filename for the curve 2
filenames_pos2 = ['cam2_001.h5', 'cam2_002.h5', 'cam2_003.h5',
                  'cam2_004.h5', 'cam2_005.h5', 'cam2_006.h5',
                  'cam2_007.h5', 'cam2_008.h5', 'cam2_009.h5',
                  'cam2_010.h5', 'cam2_011.h5', 'cam2_012.h5',
                  'cam2_013.h5', 'cam2_014.h5', 'cam2_015.h5',
                  'cam2_016.h5', 'cam2_017.h5', 'cam2_018.h5',
                  'cam2_019.h5', 'cam2_020.h5', 'cam2_021.h5',
                  'cam2_022.h5', 'cam2_023.h5']

# Voltage for the curve 1
gain_1 = [
    400, 500, 600, 700, 800, 900, 1000, 1100,
    1150, 1200, 1250, 1275, 1300, 1325, 1350,
    1370, 1380, 1390, 1400, 1410, 1420, 1430,
    1440, 1450, 1460, 1470, 1480, 1490, 1500,
    1510]

# Voltage for the curve 2
gain_2 = [600, 700, 800, 900, 1000, 1100, 1150, 1175, 1200, 1225,
          1250, 1275, 1300, 1325, 1350, 1360, 1370, 1380, 1390, 1400,
          1410, 1420, 1430]

# Dark frame, no MCP no UV light (may not be used)
dark_frame = 'cam2_000.h5'

# Level frame, low MCP voltage and UV light on
level_frame = ['cam2_001.h5', 'cam2_001.h5']

# Concat filenames
filenames = [filenames_pos1, filenames_pos2]

# Concate gain
gainsv = [gain_1, gain_2]

# Plot color
plot_c = ['#0C5DA5', '#00B945', '#FF9500',
          '#FF2C00', '#845B97', '#474747', '#9e9e9e']

# Curve fit first point on the ith file.
fit_i = [18, 15]

# The curve is computed in this square
id_square = 8

def main():
    """ 
    This script create two gain curve from two set of data.
    """
    #plt.style.use(['science', 'ieee'])
    fig, axs = plt.subplots(figsize=(10, 6))

    # Set label and color
    axs.set_prop_cycle(cycler(
        'color', ['r', 'g', 'b', 'y']) + cycler('linestyle', ['-', '--', ':', '-.']))
    pos = ["MCP Y", "MCP X"]

    # Create squares and roi arrays
    squares = gen_grid_square((20, 20), (9, 5), (50, 50), (60, 60))
    roi = np.array([[180, 130], [780, 450]])

    # Loop of over the data set
    for idx, files in enumerate(filenames):
        signals = []
        data, _, _, _ = nu.load_images(BASE[idx] + level_frame[idx])
        img_level = nc.roi_image(np.mean(data, axis=0), roi)
        gain = gainsv[idx]

        # Loop over file
        for file in files:
            data, _, _, _ = nu.load_images(
                BASE[idx] + file)
            img = nc.roi_image(np.mean(data, axis=0), roi)/img_level
            means = calc_means_of_squares(img, squares)
            signals.append(means)
            print(means[id_square])

        signals = np.asarray(signals)

        vmcpfit = np.linspace(1400, 2400)
        popt, pcov = curve_fit(
            func_lin, gain[fit_i[idx]:-1], np.log(signals[fit_i[idx]:-1, id_square]))

        axs.semilogy(gain, signals[:, id_square],
                     label='{} data'.format(pos[idx]), color=plot_c[idx], marker='.', ls='None')
        axs.semilogy(vmcpfit, func_exp(vmcpfit, *popt),
                     label='{} extrapolation'.format(pos[idx]), color=plot_c[idx], ls='-')
        axs.axvline(2000, label='G(2000V)={}'.format(
            func_exp(2000, *popt)), color=plot_c[idx])

    axs.set_title('Gain curve MCP')
    axs.set_xlabel('MCP voltage (V)')
    axs.set_ylabel('Camera signal (AU)')
    axs.set_xlim((900, 2400))
    axs.grid(True, axis="both", which="both")
    axs.legend()
    plt.show()


def calc_means_of_squares(image, squares):
    means = []
    for idx, sq in enumerate(squares):
        temp = np.mean(image[sq.y[0]:sq.y[1], sq.x[0]:sq.x[1]])
        means.append(temp)
    return means


def func_lin(x, a, b):
    return a * x + b


def func_exp(x, a, b):
    return np.exp(a*x)*np.exp(b)


def gen_grid_square(offset, n, size, span):
    squares = []
    for y in range(n[1]):
        for x in range(n[0]):
            xbase = offset[0] + x * span[0]
            ybase = offset[1] + y * span[1]
            sq = Square([xbase, xbase + size[0]], [ybase, ybase + size[1]])
            squares.append(sq)

    return squares


def plot_square(axe, squares):
    for idx, sq in enumerate(squares):
        req = Rectangle((sq.x[0], sq.y[0]),
                        sq.size[0], sq.size[1], fill=False, edgecolor='red')
        axe.add_artist(req)


class Square:
    """
    docstring
    """

    def __init__(self, x1, x2, y1, y2):
        self.x = [x1, x2]
        self.y = [y1, y2]
        self.__calc_size()

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.__calc_size()

    def __calc_size(self):
        self.size = [self.x[1] - self.x[0], self.y[1] - self.y[0]]


if __name__ == "__main__":
    main()
