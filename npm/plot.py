import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import numpy as np
from pytz import timezone

import npm.proc as nc
import npm.utils as nu


class BasePlot:
    """
    Base class for plotting data.
    """
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24
    BIGGEST_SIZE = 28

    def __init__(self, xs=9, ys=6, title='Global title'):
        self.fig = plt.figure(figsize=[xs, ys])
        #plt.rcParams['backend'] = 'pdf'
        #self.fig.suptitle(title, fontsize=self.BIGGEST_SIZE)
        self.axes = []
        # plt.rc('font', size=self.SMALL_SIZE)  # controls default text sizes
        # plt.rc('axes', titlesize=self.SMALL_SIZE)  # fontsize of the axes title
        # plt.rc('axes', labelsize=self.MEDIUM_SIZE)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=self.SMALL_SIZE)  # fontsize of the tick labels
        # plt.rc('legend', fontsize=self.SMALL_SIZE)  # legend fontsize
        # plt.rc('figure', titlesize=self.BIGGER_SIZE)  # fontsize of the figure title
        plt.style.use('science_soutenance')
        #plt.tight_layout()

    def draw(self):
        plt.draw()

    def show(self):
        plt.show()

    def close(self):
        plt.close()

    def legend(self, loc=0):
        plt.legend(loc=loc)

    def savefig(self, filename):
        plt.savefig(filename, bbox_inches='tight')

    def axis_time(self, time):
        date_fmt = '%H:%M:%S'
        date_formatter = mdate.DateFormatter(
            date_fmt, tz=timezone('Europe/Paris'))
        for axe in self.axes:
            axe.xaxis.set_major_formatter(date_formatter)
        return mdate.epoch2num(time)

    def grid(self):
        for axe in self.axes:
            axe.grid()

    def hold(self):
        plt.h

    def setBoxColors(self, bp):
        plt.setp(bp['boxes'][0], color='blue')
        plt.setp(bp['caps'][0], color='blue')
        plt.setp(bp['caps'][1], color='blue')
        plt.setp(bp['whiskers'][0], color='blue')
        plt.setp(bp['whiskers'][1], color='blue')
        #plt.setp(bp['fliers'][0], color='blue')
        #plt.setp(bp['fliers'][1], color='blue')
        plt.setp(bp['medians'][0], color='blue')

        plt.setp(bp['boxes'][1], color='red')
        plt.setp(bp['caps'][2], color='red')
        plt.setp(bp['caps'][3], color='red')
        plt.setp(bp['whiskers'][2], color='red')
        plt.setp(bp['whiskers'][3], color='red')
        #plt.setp(bp['fliers'][2], color='red')
        #plt.setp(bp['fliers'][3], color='red')
        plt.setp(bp['medians'][1], color='red')


class SimplePlot(BasePlot):
    def __init__(self, xs=9, ys=6, title='Global title'):
        BasePlot.__init__(self, xs, ys, title)
        # [0.1, 0.1, 0.80, 0.80]
        self.axes.append(self.fig.add_subplot(111))


class SimpleTwinPlot(SimplePlot):
    def __init__(self, xs=9, ys=6, title='Global title'):
        SimplePlot.__init__(self, xs, ys, title)
        self.axes.append(self.axes[0].twinx())


class DualPlot(BasePlot):
    """
    Two plot in same figure.
    """
    def __init__(self, xs=9, ys=6, title='Global title', vertical=False):
        BasePlot.__init__(self, xs, ys, title)
        if vertical:
            self.axes.append(self.fig.add_axes([0.1, 0.1, 0.80, 0.35]))
            self.axes.append(self.fig.add_axes([0.1, 0.55, 0.80, 0.35]))
        else:
            self.axes.append(self.fig.add_axes([0.1, 0.1, 0.35, 0.80]))
            self.axes.append(self.fig.add_axes([0.55, 0.1, 0.35, 0.80]))


class DualTwinPlot(DualPlot):
    def __init__(self, xs=9, ys=6, title='Global title', vertical=False):
        DualPlot.__init__(self, xs, ys, title, vertical)
        self.axes.append(self.axes[0].twinx())
        self.axes.append(self.axes[1].twinx())


class QuadPlot(BasePlot):
    def __init__(self, xs=9, ys=6, title='Global title',):
        BasePlot.__init__(self, xs, ys, title)
        self.axes.append(self.fig.add_axes([0.1, 0.1, 0.35, 0.35]))
        self.axes.append(self.fig.add_axes([0.1, 0.55, 0.35, 0.35]))
        self.axes.append(self.fig.add_axes([0.55, 0.1, 0.35, 0.35]))
        self.axes.append(self.fig.add_axes([0.55, 0.55, 0.35, 0.35]))


def profile(image, x_axis=None, y_axis=None):
    use_pcolor = True
    if x_axis is None:
        x_axis = np.arange(image.shape[1])
        use_pcolor = False
    if y_axis is None:
        y_axis = np.arange(image.shape[0])
        use_pcolor = False

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('Profils et projections')
    ax1 = fig.add_axes([0.1, 0.4, 0.5, 0.5])
    ax2 = fig.add_axes([0.1, 0.1, 0.5, 0.2])
    ax3 = fig.add_axes([0.7, 0.4, 0.2, 0.5])

    if use_pcolor == True:
        ax1.pcolor(x_axis, y_axis, image)
        ax1.set_xlabel('y (mm)')
        ax1.set_ylabel('z (mm)')
        ax2.set_xlabel('y plane (mm)')
        ax3.set_ylabel('z plane (mm)')
    else:
        ax1.imshow(image, aspect='auto', clim=(1000, 10000))
        ax1.set_xlabel('y (px)')
        ax1.set_ylabel('z (px)')
        ax2.set_xlabel('y plane (px)')
        ax3.set_ylabel('z plane (px)')

    ax2.plot(x_axis, image.mean(0))

    ax3.plot(image.mean(1), y_axis)
    ax3.invert_yaxis()
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position('right')

    # Titres
    ax1.set_title('Image MCP')
    ax2.set_title('Projection y')

    ax2.set_ylabel('Projection (moyenne cps)')
    ax3.set_title('Projection z')
    ax3.set_xlabel('Projection (moyenne cps)')

    plt.draw()
    return None


def image(image, x_axis=None, y_axis=None):
    use_pcolor = True
    if x_axis is None:
        x_axis = np.arange(image.shape[1])
        use_pcolor = False
    if y_axis is None:
        y_axis = np.arange(image.shape[0])
        use_pcolor = False

    profil = image.mean(0)
    popt, _ = nc.fit_beam(profil, x_axis, nc.double_gaussian_norm)
    popt1 = np.append(0, popt[1:4])
    popt2 = np.append(popt[0], popt[4:7])

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('Beam profile measurement in the Y plane with the symmetric IPM, $\pm 5kV$')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    if use_pcolor == True:
        ax1.imshow(image, cmap=plt.cm.get_cmap('gray'), interpolation='none',
                   extent=[x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()], clim=(4000, 20000))
        ax1.set_xlabel('y (mm)')
        ax1.set_ylabel('z (mm)')
        ax2.set_xlabel('y (mm)')
        ax2.set_ylabel('Counts')
    else:
        ax1.imshow(image, aspect='auto', clim=(1000, 10000))
        ax1.set_xlabel('y (px)')
        ax1.set_ylabel('z (px)')
        ax2.set_xlabel('y (px)')
        ax2.set_ylabel('Counts')

    plt.plot(x_axis, profil, 'b', label='Signal')

    ax2.plot(x_axis, nc.gaussian_norm(x_axis, *popt2), 'g-.',
             label='Fit g1: $\mu = {0:.2f}mm$ $\sigma = {1:.2f}mm$'.format(popt2[2], popt2[3]))
    ax2.plot(x_axis, nc.gaussian_norm(x_axis, *popt1) + popt[0], 'g:',
             label='Fit g2: $\mu = {0:.2f}mm$ $\sigma = {1:.2f}mm$'.format(popt1[2], popt1[3]))

    ax2.plot(x_axis, nc.gaussian_norm(x_axis, *popt1) + nc.gaussian_norm(x_axis, *popt2), 'r--', label='g1 + g2')

    # Titres
    ax1.set_title('Raw image of one beam pulse')
    ax2.set_title('Profile projection by averaging image in z direction')

    plt.legend(loc=0)
    plt.draw()

    return None

def time_bpm(image, time, x_axis=None, y_axis=None):
    if x_axis is None:
        x_axis = np.arange(image.shape[1])
    if y_axis is None:
        y_axis = np.arange(image.shape[0])

    secs_bpm6y, vals_bpm6y = nu.getDataFromArchiver(
        'LHE:Y_BPM_6', nu.SCALAR, nu.epics_time_str(time[0] - 688), nu.epics_time_str(time[-1] - 688))
    npvals_bpm6y = np.asarray(vals_bpm6y)
    secsnp_bpm6y = np.asarray(secs_bpm6y) + 688
    secsnp_bpm6y = mdate.epoch2num(secsnp_bpm6y)

    popt = nc.fit_beam2(image, x_axis, nc.double_gaussian_norm)

    time_x = mdate.epoch2num(np.asarray(time))
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    date_fmt = '%H:%M:%S'
    date_formatter = mdate.DateFormatter(
        date_fmt, tz=timezone('Europe/Paris'))
    ax1.xaxis.set_major_formatter(date_formatter)

    ax1.set_title('Beam position comparison')
    ax1.set_ylabel('Beam position (mm)')
    ax1.set_xlabel('Time')

    ax1.plot(time_x, popt[:, 2], 'g-.', label='Beam halo')
    ax1.plot(time_x, popt[:, -2], 'g:', label='Beam core')
    ax1.plot(secsnp_bpm6y, npvals_bpm6y[:, 0], 'r', label='BPM')

    fig.autofmt_xdate()
    plt.legend()
    plt.draw()

    return None

def time_sigma(image, time, x_axis=None, y_axis=None):
    if x_axis is None:
        x_axis = np.arange(image.shape[1])
    if y_axis is None:
        y_axis = np.arange(image.shape[0])

    popt = nc.fit_beam2(image, x_axis, nc.double_gaussian_norm)

    time_x = mdate.epoch2num(np.asarray(time))


    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    date_fmt = '%H:%M:%S'
    date_formatter = mdate.DateFormatter(
        date_fmt, tz=timezone('Europe/Paris'))
    ax1.xaxis.set_major_formatter(date_formatter)

    ax1.set_title('Beam position comparison')
    ax1.set_ylabel('Beam position (mm)')
    ax1.set_xlabel('Time')

    ax1.plot(time_x, popt[:, 3], 'g-.', label='Beam halo')
    ax1.plot(time_x, popt[:, -1], 'g:', label='Beam core')

    fig.autofmt_xdate()
    plt.legend()
    plt.draw()

    return None