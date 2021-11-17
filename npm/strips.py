from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import concurrent.futures
from multiprocessing import current_process, cpu_count, Process
from operator import concat
import threading

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import uproot3
from scipy import optimize
from scipy.interpolate import interp1d

weight_gaus = np.asarray((1, 1, 1, 1, 1, 1, 1, 9, 5, 3, 2, 1.5, 1, 0.9,
                          0.8, 0.8, 0.8, 0.8, 0.9, 1, 1.5, 2, 3, 5, 9, 1, 1, 1, 1, 1, 1, 1))
pos_gaus = np.asarray((20.54, 13.4, 8.28, 6.66, 4.79, 3.42, 2.35, 1.38, 0.46))
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
            profile = self.peaks[i]

            

            # f1 = interp1d(np.pad(self.x_raw_gaus, (1, 1), 'linear_ramp', end_values=(-20, 20)),
            #              np.pad(profile, (1, 1), 'constant', constant_values=(0, 0)), kind='nearest')
            p0 = [np.max(self.peaks[i, :]),
                  self.x_raw_lin[np.argmax(self.peaks[i, :])], 3]

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
        """
        bins = np.zeros(32 * 2)
        bins[::2] += 0.120
        bins[1::2] += 0.8
        bins = np.insert(np.cumsum(bins), 0, 0)
        self.bins = bins - np.max(bins) / 2

    def _construct_bin_axis_gauss(self):
        binsg = np.zeros(18 * 2)
        inter = np.diff(pos_gaus) + (size_gaus/2 +
                                     np.roll(size_gaus/2, -1))[:-1]
        inter = np.abs(np.concatenate([inter, inter[::-1]]))
        inter = np.pad(inter, (1, 1), constant_values=0.1)
        binsg[::2] += inter
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

    def normalize(self):
        self.peaks = self.peaks


class FindPeak(object):
    def __init__(self, signal):
        self.signal = signal
        self.signal_mean = np.mean(self.signal, axis=0)
        self.signal_std = np.std(self.signal, axis=0)
        self.pulses = []

    def threshold(self, level=3, multi=5):
        self.map_thres = (self.signal > (self.signal_mean +
                                         level * self.signal_std)).astype(int)
        self.multiplicity = (
            np.sum(self.map_thres, axis=1) > multi).astype(int)
        self.find_consecutive(self.multiplicity)
        self.construct_pulses()

    def find_consecutive(self, a):
        iszero = np.concatenate(([0], np.equal(a, 1).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        self.ranges = np.where(absdiff == 1)[0].reshape(-1, 2)

    def construct_pulses(self, consec=5):
        self.pulses.clear()
        for x in self.ranges:
            if x[1] - x[0] > consec:
                data = self.signal[x[0]:x[1], :]
                pulse = np.nansum(np.where(data>=0, data, np.nan), axis=0)
                # pulse = np.sum(self.signal[x[0]:x[1], :], axis=0)
                # print(pulse)
                self.pulses.append(pulse)
        self.pulses = np.asarray(self.pulses)


class ProcFaster(object):
    def __init__(self, filename, card=1, dt=20e-6, n_int=2):
        self.filename = filename
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts
        file = uproot3.open(filename)
        self.tree = file['card' + str(card) + '_tree']
        self.raw_charge = self.tree['charge'].array()
        self.time = self.tree['time'].array()
        self.N = self.time.size
        self.fft_m = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.fft_cut = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.sig_filter = np.zeros(np.shape(self.raw_charge))

    def remove_pedestal(self):
        strip_mean = np.mean(self.raw_charge, axis=0)
        self.sub_signal = self.raw_charge[:] - strip_mean

    def normalize_strips(self):
        self.sig_filter = self.sig_filter/weight_gaus

    def run_fft(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            res = {executor.submit(
                self.__perform_fft, self.sub_signal[:, x], x): x for x in range(32)}
            for future in concurrent.futures.as_completed(res):
                url = res[future]
        # processes = [threading.Thread(target=self.__perform_fft, args=(
        #     self.sub_signal[:, x], x)) for x in range(32)]
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()

    def run_ifft(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            res = {executor.submit(
                self.__perform_ifft, self.fft_cut[:, x], x): x for x in range(32)}
            for future in concurrent.futures.as_completed(res):
                url = res[future]
        # processes = [threading.Thread(target=self.__perform_ifft, args=(
        #     self.fft_cut[:, x], x)) for x in range(32)]
        # for p in processes:
        #     p.start()
        # for p in processes:
        #     p.join()

    def __perform_fft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.fft_m[:, x] = pyfftw.interfaces.numpy_fft.fft(a)

    def __perform_ifft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.sig_filter[:, x] = np.real(pyfftw.interfaces.numpy_fft.ifft(a))

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
        Fstep = self.Fs / self.N
        Ncut = int(fcut / Fstep)
        self.fft_cut = np.array(self.fft_m)
        self.fft_cut[Ncut:-1 - Ncut, :] = 0
        pass

    def high_filter(self):
        pass

# Quelque modifications pour fonctionner avec le convertisseur de Francesca.


class ProcFaster2(object):
    def __init__(self, filename, dt=10e-6, n_int=2, card=1):
        self.filename = filename
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts
        file = uproot3.open(filename)
        self.tree = file['DataTree']
        self.card = self.tree['card'].array()
        self.raw_charge = self.tree['Q'].array()
        self.time = self.tree['time'].array()
        self.raw_charge = self.raw_charge[self.card == card]
        self.time = self.time[self.card == card]
        self.N = self.time.size
        self.fft_m = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.fft_cut = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.sig_filter = np.zeros(np.shape(self.raw_charge))

    def remove_pedestal(self):
        strip_mean = np.mean(self.raw_charge, axis=0)
        self.sub_signal = self.raw_charge[:] - strip_mean

    def normalize_strips(self):
        self.sig_filter = self.sig_filter/weight_gaus

    def run_fft(self):
        processes = [threading.Thread(target=self.__perform_fft, args=(
            self.sub_signal[:, x], x)) for x in range(32)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def run_ifft(self):
        processes = [threading.Thread(target=self.__perform_ifft, args=(
            self.fft_cut[:, x], x)) for x in range(32)]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def __perform_fft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.fft_m[:, x] = pyfftw.interfaces.numpy_fft.fft(a)

    def __perform_ifft(self, signal, x):
        N = np.size(signal)
        a = pyfftw.empty_aligned(N, dtype='complex128', n=16)
        a[:] = signal
        self.sig_filter[:, x] = np.real(pyfftw.interfaces.numpy_fft.ifft(a))

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
        Fstep = self.Fs / self.N
        Ncut = int(fcut / Fstep)
        self.fft_cut = np.array(self.fft_m)
        self.fft_cut[Ncut:-1 - Ncut, :] = 0
        pass

    def high_filter(self):
        pass


class ProcFaster3(object):
    def __init__(self, filename, dt=10e-6, n_int=2, card=1):
        self.filename = filename
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts
        file = uproot3.open(filename)
        tree = file['card' + str(card) + '_tree']
        self.raw_charge = tree['charge'].array()
        self.time = tree['time'].array()
        self.N = self.time.size
        self.fft_m = np.zeros(np.shape(self.raw_charge), dtype=complex)
        #self.sig_filter = np.zeros(np.shape(self.raw_charge))

    def remove_pedestal(self):
        strip_mean = np.mean(self.raw_charge, axis=0)
        self.raw_charge = self.raw_charge[:] - strip_mean

    def normalize_strips(self):
        self.raw_charge = self.raw_charge/weight_gaus

    def run_fft(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            res = {executor.submit(
                self.__perform_fft, self.raw_charge[:, x], x): x for x in range(32)}
            for future in concurrent.futures.as_completed(res):
                url = res[future]

    def run_ifft(self):
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
        Fstep = self.Fs / self.N
        Ncut = int(fcut / Fstep)
        # self.fft_cut = np.array(self.fft_m)
        self.fft_m[Ncut:-1 - Ncut, :] = 0
        pass

    def high_filter(self):
        pass
