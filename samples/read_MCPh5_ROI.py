import npm.utils as nu
import npm.proc as nc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Change variable to be compliant with your setup.
BASE = '/local/home/fb250757/Documents/Data/Camera/cam2/'
FILENAME = '2022_06_11/cam2_010.h5'

def main():
    """ This script load h5 images from a file, create a ROI on it and show the resulting image.
    """
    # Create a figure
    fig, axs = plt.subplots(3, 1, figsize=(6, 10))

    # Load images
    data_X, _, _, _ = nu.load_images(BASE + FILENAME)
    image = np.mean(data_X, axis=0)

    # ROI array [[x_start, y_start], [x_stop, y_stop]]
    roi = np.array([[110, 150], [810, 490]])
    # Apply ROI
    image_roi = nc.roi_image(image, roi)

    # Create a square array.
    squares = gen_grid_square((50, 20), (10, 5), (50, 50), (60, 60))
    # Plot the squares on the image.
    plot_square(axs[1], squares)
    # Compute the mean value in the squares
    mean_sq = calc_means_of_squares(image_roi, squares)

    # Plotting block
    axs[0].imshow(image) # The image
    axs[1].imshow(image_roi) # The ROI
    axs[2].plot(mean_sq.reshape(10, 5)) # The mean in the squares
    plt.show()


def calc_means_of_squares(image, squares):
    """ Compute the mean value inside each Square in the input List.

    Args:
        image (image): Image data.
        squares (list): List of squares

    Returns:
        list: List of mean values.
    """
    means = []
    for idx, sq in enumerate(squares):
        temp = np.mean(image[sq.y[0]:sq.y[1], sq.x[0]:sq.x[1]])
        means.append(temp)
    return np.asarray(means)


def gen_grid_square(offset, n, size, span):
    """ Generate a list of square

    Args:
        offset (list): starting point (x,y) of the square grid
        n (list): number (x,y) of the squares in the grid
        size (list): Size of the grid (x,y)
        span (list): Span between two squares (x,y)

    Returns:
        list: List of Squares
    """
    squares = []
    for y in range(n[1]):
        for x in range(n[0]):
            xbase = offset[0] + x * span[0]
            ybase = offset[1] + y * span[1]
            sq = Square([xbase, xbase + size[0]], [ybase, ybase + size[1]])
            squares.append(sq)

    return squares


def plot_square(axe, squares):
    """ Plot the square on an image

    Args:
        axe (matplotlib axe): Axe
        squares (list): List of the squares to plot.
    """
    for idx, sq in enumerate(squares):
        req = Rectangle((sq.x[0], sq.y[0]), sq.size[0],
                        sq.size[1], fill=False, edgecolor='red')
        axe.add_artist(req)
        axe.annotate(str(idx), (sq.x[0]+sq.size[0]/2, sq.y[0]+sq.size[1]/2), color='w', weight='bold',
                     fontsize=8, ha='center', va='center')


class Square:
    """ Square class.
    This class contaons only the square coordinates and some helper functions.
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
