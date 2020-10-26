import threading

import matplotlib.pyplot as plt
import numpy as np
import pyfftw
import uproot
from scipy import optimize
from scipy.interpolate import interp1d


class FormatPeak(object):
    weight_gaus = np.asarray((1, 1, 1, 1, 1, 1, 1, 9, 5, 3, 2, 1.5, 1, 0.9,
                              0.8, 0.8, 0.8, 0.8, 0.9, 1, 1.5, 2, 3, 5, 9, 1, 1, 1, 1, 1, 1, 1))
    pos_gaus = np.asarray((13.4, 8.28, 6.66, 4.79, 3.42, 2.35, 1.38, 0.46))
    weight_lin = np.asarray((1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1))

    def __init__(self, peaks, gaussian=False):
        self.peaks = peaks
        self.gaussian = gaussian
        self._construct_bin_axis_lin()
        self._construct_axis_lin()
        self._construct_axis_gauss()
        self._construct_bin_signal_lin()
        # self.fit_beam()

    def fit_beam(self):
        self.popt = np.zeros((self.peaks.shape[0], 3))
        for i in range(0, self.peaks.shape[0]):
            profile = self.peaks[i]

            x_fit = np.linspace(-20, 20, 400)

            f1 = interp1d(np.pad(self.x_raw_lin, (1, 1), 'linear_ramp', end_values=(-20, 20)),
                          np.pad(profile, (1, 1), 'constant', constant_values=(0, 0)), kind='nearest')

            # fit_profile = np.pad(profile, (32,32), 'constant', constant_values=(0, 0))
            # fit_x = np.pad(self.x_raw_lin, (32, 32), 'linear_ramp', end_values=(-40, 40))

            #plt.plot(x_fit, f1(x_fit))np.min(profile),
            p0 = [np.max(profile), self.x_raw_lin[np.argmax(profile)], 3]#, np.max(profile) / 4, self.x_raw_lin[np.argmax(profile)], 6]
            try:
                self.popt[i], pcov = optimize.curve_fit(self.gaussian_norm, x_fit, f1(x_fit), p0=p0)
            except RuntimeError:
                pass
                # print('Fit failed !')
            #plt.plot(x_fit, self.double_gaussian_norm(x_fit, *self.popt[i]))
            #plt.show()

    def double_gaussian_norm(self, x, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
        return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2)) + amplitude2 / np.sqrt(
            stddev2) * np.exp(-(x - mean2) ** 2 / (2. * stddev2 ** 2))

    def gaussian_norm(self, x, amplitude1, mean1, stddev1):
        return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2))

    def _construct_bin_axis_lin(self):
        bins = np.zeros(32 * 2)
        bins[::2] += 0.120
        bins[1::2] += 0.8
        bins = np.insert(np.cumsum(bins), 0, 0)
        self.bins = bins - np.max(bins) / 2

    def _construct_bin_signal_lin(self):
        self.s_pulses = np.dstack((self.peaks, np.zeros_like(
            self.peaks))).reshape(self.peaks.shape[0], -1)
        self.s_pulses = np.insert(self.s_pulses, 0, values=0, axis=1)

    def _construct_axis_lin(self):
        x_raw_lin = np.arange(32) * 0.92
        self.x_raw_lin = x_raw_lin - x_raw_lin[15] - 0.92 / 2

    def _construct_axis_gauss(self):
        x_raw_gaus = np.append(self.pos_gaus, -np.flip(self.pos_gaus))
        self.x_raw_gaus = np.pad(-x_raw_gaus, (7, 7), 'linear_ramp',
                                 end_values=(self.x_raw_lin[0], self.x_raw_lin[-1]))

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
        self.multiplicity = (np.sum(self.map_thres, axis=1) > multi).astype(int)
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
                pulse = np.sum(self.signal[x[0]:x[1], :], axis=0)
                # print(pulse)
                self.pulses.append(pulse)
        self.pulses = np.asarray(self.pulses)


class ProcFaster(object):
    def __init__(self, filename, dt=10e-6, n_int=2):
        self.filename = filename
        self.dt = dt
        self.n_int = n_int
        self.Ts = self.dt * self.n_int
        self.Fs = 1 / self.Ts
        file = uproot.open(filename)
        self.tree = file['card2_tree']
        self.raw_charge = self.tree['charge'].array()
        self.time = self.tree['time'].array()
        self.N = self.time.size
        self.fft_m = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.fft_cut = np.zeros(np.shape(self.raw_charge), dtype=complex)
        self.sig_filter = np.zeros(np.shape(self.raw_charge))

    def remove_pedestal(self):
        strip_mean = np.mean(self.raw_charge, axis=0)
        self.sub_signal = self.raw_charge[:] - strip_mean

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
