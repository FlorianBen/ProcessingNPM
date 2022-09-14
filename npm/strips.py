import concurrent.futures

import numpy as np
import pyfftw
import uproot3
from scipy import optimize

weight_gaus = np.asarray((1, 1, 1, 1, 1, 1, 1, 9, 5, 3, 2, 1.5, 1, 0.9,
                          0.8, 0.8, 0.8, 0.8, 0.9, 1, 1.5, 2, 3, 5, 9, 1, 1, 1, 1, 1, 1, 1))
pos_gaus = np.asarray((20.52, 13.4, 9.28, 6.66, 4.79, 3.42, 2.35, 1.38, 0.46))
size_gaus = np.asarray((9, 5, 3, 2, 1.5, 1, 0.9, 0.8, 0.8))
weight_lin = np.asarray((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))


class FormatPeak(object):
    """
    Helper class for plotting and fitting the beam.
    """

    def __init__(self, peaks, gaussian=False):
        """
        Constructor

        Args:
            peaks (ndarray): Array of peaks.
            gaussian (bool, optional): Use gaussian strips. Defaults to False.
        """
        self.peaks = peaks
        self.gaussian = gaussian
        self._construct_bin_axis_lin()
        self._construct_bin_axis_gauss()
        self._construct_axis_lin()
        self._construct_axis_gauss()
        self._construct_bin_signal_lin()

    def fit_beam(self):
        """
        Fit the beam using a least square optimization.
        """
        self.popt = np.zeros((self.peaks.shape[0], 3))
        for i in range(0, self.peaks.shape[0]):

            if self.gaussian:
                average = np.average(self.x_raw_gaus, weights=self.peaks[i, :])
                variance = np.average(
                    (self.x_raw_gaus-average)**2, weights=self.peaks[i, :])
            else:
                average = np.average(self.x_raw_lin, weights=self.peaks[i, :])
                variance = np.average(
                    (self.x_raw_lin-average)**2, weights=self.peaks[i, :])

            p0 = [np.max(self.peaks[i, :]),
                  self.x_raw_lin[np.argmax(self.peaks[i, :])], np.sqrt(variance)]

            if self.gaussian:
                self.x_fit = self.x_raw_gaus
            else:
                self.x_fit = self.x_raw_lin

            try:
                self.popt[i], pcov = optimize.curve_fit(
                    self.gaussian_norm, self.x_fit, self.peaks[i, :], p0=p0)
            except RuntimeError:
                print('Fit failed !')
                pass

    def fwhm_beam(self):
        """ 
            Compute the fwhm on each bunch.
        """
        self.fwhm_info = np.zeros((self.peaks.shape[0], 4))
        for i in range(0, self.peaks.shape[0]):
            profile = self.peaks[i]
            y_min = np.min(profile)
            y_max = np.max(profile)
            i_max = np.argmax(profile)
            dr = y_max - y_min
            x_max = self.x_raw_lin[i_max]

            fwhm_val = dr / 2 + y_min

            a_conv = True
            b_conv = True

            a_i = 1
            b_i = -1

            while a_conv or b_conv:
                a = profile[i_max + a_i]
                b = profile[i_max + b_i]

                if a - y_min > dr / 2:
                    a_i = a_i + 1
                else:
                    a_conv = False

                if b - y_min > dr / 2:
                    b_i = b_i - 1
                else:
                    b_conv = False
            if i_max + b_i < 0:
                low = 0
            else:
                low = i_max + b_i
            if i_max + a_i > (self.x_raw_lin.size - 1):
                up = (self.x_raw_lin.size() - 1)
            else:
                up = i_max + a_i
            fwhm = self.x_raw_lin[low:i_max + a_i]
            self.fwhm_info[i] = np.asarray(
                [fwhm[0], fwhm[-1], fwhm_val, x_max])

    def double_gaussian_norm(self, x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
        """
        Fitting function double gaussian.

        Args:
            x (double): Position
            amplitude1 (double): 1rst Gaussian amplitude
            mean1 (double): 1rst Gaussian position
            stddev1 (double): 1rst Gaussian std
            amplitude2 (double): 2nd Gaussian amplitude
            mean2 ([type]): 2rst Gaussian position
            stddev2 ([type]): 2rst Gaussian std

        Returns:
            [double]: Double gaussian value
        """
        return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2)) + amplitude2 / np.sqrt(
            stddev2) * np.exp(-(x - mean2) ** 2 / (2. * stddev2 ** 2))

    def gaussian_norm(self, x, amplitude1, mean1, stddev1):
        """
        Fitting function gaussian.

        Args:
            x (double): Position
            amplitude1 (double): Gaussian amplitude
            mean1 (double): Gaussian position
            stddev1 (double): Gaussian std
        Returns:
            [double]: Gaussian value
        """
        return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2))

    def _construct_bin_axis_lin(self):
        """
        Construct the axis for linear strip.
        Strips are 800 um wide and spaced by 120 um.
        """
        bins = np.zeros(32 * 2)
        bins[::2] += 0.120
        bins[1::2] += 0.8
        bins = np.insert(np.cumsum(bins), 0, 0)
        self.bins = bins - np.max(bins) / 2

    def _construct_bin_axis_gauss(self):
        """
        Construct the axis for gaussian strips.
        These strips have different size but the span between 2 strips is equal to 120 um
        """
        binsg = np.zeros(18 * 2)
        binsg[::2] += 0.120
        binsg[1::2] = np.concatenate([size_gaus, size_gaus[::-1]])
        binsg = np.insert(np.cumsum(binsg), 0, 0)
        self.binsg = binsg - np.max(binsg) / 2

    def _construct_bin_signal_lin(self):
        self.s_pulses = np.dstack((self.peaks, np.zeros_like(
            self.peaks))).reshape(self.peaks.shape[0], -1)
        self.s_pulses = np.insert(self.s_pulses, 0, values=0, axis=1)

    def _construct_bin_signal_gaus(self):
        self.s_pulses = np.dstack((self.peaks, np.zeros_like(
            self.peaks))).reshape(self.peaks.shape[0], -1)
        self.s_pulses = np.insert(self.s_pulses, 0, values=0, axis=1)

    def _construct_axis_lin(self):
        x_raw_lin = np.arange(32) * 0.92
        self.x_raw_lin = x_raw_lin - x_raw_lin[15] - 0.92 / 2

    def _construct_axis_gauss(self):
        self.x_raw_gaus = np.append(pos_gaus, -np.flip(pos_gaus))


class FindPeak(object):
    def __init__(self, signal, dt=10*10e-6, n_int=2):
        """Find peak object will search pulse in the raw signal.

        The search algorithm is based on amplitude and temporal thresholds. 
        If multiple strips are above a given during n time steps then a pulse is detected.
        The signal is integrated during the detected window.   
        Args:
            signal (ndarray): Input signal
        """
        self.signal = signal
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts
        self.signal_mean = np.mean(self.signal, axis=0)
        self.signal_std = np.std(self.signal, axis=0)
        self.pulses = []

    def threshold(self, level=3.0, multi_strip=5):
        """ Set the threshold values and launch the search algorithm.

        Args:
            level (float, optional): Factor on the amplitude threshold based on one STD. Defaults to 3.0.
            multi_strip (float, optional): Multiplicity factor on strips. Defaults to 5.0.
        """
        self.map_thres = (self.signal > (self.signal_mean +
                                         level * self.signal_std)).astype(int)
        self.multiplicity = (
            np.sum(self.map_thres, axis=1) > multi_strip).astype(int)
        self.find_consecutive(self.multiplicity)
        self.construct_pulses()

    def find_consecutive(self, multi_strip):
        """ Search for consequitive triggered strip.

        Args:
            multi_time (int):  Multiplicity strips.
        """
        iszero = np.concatenate(
            ([0], np.equal(multi_strip, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        self.ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    def construct_pulses(self, consec=5):
        """ Construct the pulse array.

        Args:
            consec (int, optional): Number of consecutive triggers. Defaults to 5.
        """
        self.pulses.clear()
        for x in self.ranges:
            if x[1] - x[0] > consec:
                data = self.signal[x[0]:x[1], :]
                pulse = np.nansum(np.where(data >= 0, data, np.nan), axis=0)
                self.pulses.append(pulse)
        self.pulses = np.asarray(self.pulses)


class ProcFaster3(object):
    def __init__(self, filename, card=1, dt=10*10e-6, n_int=2):
        self.filename = filename
        file = uproot3.open(filename)
        tree = file['card' + str(card) + '_tree']
        self.raw_charge = tree['charge'].array()
        self.time = tree['time'].array()
        self.N = self.time.size
        self.fft_m = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts

    def remove_pedestal(self):
        """ Remove background level from the signal.
        """
        strip_mean = np.mean(self.raw_charge, axis=0)
        self.raw_charge = self.raw_charge[:] - strip_mean

    def scale_std(self):
        """
        Scale each strips signal with its STD. NOT IMPLEMENTED
        """
        pass

    def normalize_strips(self):
        """ 
        Normalize the gaussian strips.
        """
        self.raw_charge = self.raw_charge/weight_gaus

    def run_fft(self):
        """
        Compute the FFT for each strips signal. 
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            res = {executor.submit(
                self.__perform_fft, self.raw_charge[:, x], x): x for x in range(32)}
            for future in concurrent.futures.as_completed(res):
                url = res[future]

    def run_ifft(self):
        """        
        Compute the inverse FFT for each FFT signal. 
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            res = {executor.submit(self.__perform_ifft,
                                   self.fft_m[:, x], x): x for x in range(32)}
            for future in concurrent.futures.as_completed(res):
                url = res[future]

    def __perform_fft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.fft_m[:, x] = pyfftw.interfaces.numpy_fft.fft(a)

    def __perform_ifft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.raw_charge[:, x] = np.real(pyfftw.interfaces.numpy_fft.ifft(a))

    def cut_freq(self, **kwargs):
        type_filter = kwargs.get('filter', 'LowPass')
        fcut = kwargs.get('fcut', self.Fs / 4)
        fband = kwargs.get('fband', self.Fs / 10)
        if type_filter == 'LowPass':
            self.low_filter(fcut)
        elif type_filter == 'HighPass':
            self.high_filter()
        else:
            pass

    def low_filter(self, fcut):
        """ 
        Cut high frequencies in the FFT signal.
        Args:
            fcut (float): Frequency cut
        """
        Fstep = self.Fs / self.N
        Ncut = int(fcut / Fstep)
        self.fft_m[Ncut:-1 - Ncut, :] = 0
        pass

    def high_filter(self):
        pass
