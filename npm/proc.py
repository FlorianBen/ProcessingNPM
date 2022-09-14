import cv2
import numpy as np
import scipy as sp
import scipy.stats
from scipy import optimize
from scipy.ndimage import interpolation, median_filter

import npm.exceptions as ne


def mean_image(images, x_axis, mag=0.2344, pixel_size=2 * 5.86e-3, axis=0):
    """ Calculate the mean image of 

    Args:
        images ([type]): [description]
        x_axis ([type]): [description]
        mag (float, optional): [description]. Defaults to 0.2344.
        pixel_size ([type], optional): [description]. Defaults to 2*5.86e-3.
        axis (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """
    popt_a = np.zeros((images.shape[0], 4))
    for i in range(0, images.shape[0]):
        popt_a[i] = fit_beam2(images[i], x_axis, gaussian_norm)

    pix = popt_a[:, 2]*mag/pixel_size
    for i in range(0, images.shape[0]):
        images[i] = np.roll(images[i], int(np.rint(pix[0] - pix[i])), axis=1)

    for i in range(0, images.shape[0]):
        popt_a[i] = fit_beam2(images[i], x_axis, gaussian_norm)

    return images.mean(axis=0)


def homography_screen(images):
    """
    Apply the homography for the third camera.
    :return:
    """
    pts_src = np.array([[242, 213], [834, 210], [853, 606], [232, 617]])
    pts_dst = np.array([[242, 213], [834, 213], [834, 509], [242, 509]])
    M, status = cv2.findHomography(pts_src, pts_dst)

    if len(images.shape) == 3:
        image_homo = []
        for i in range(0, images.shape[0]):
            h, w = images[i].shape
            image_homo.append(cv2.warpPerspective(images[i], M, (h, w)))
        return np.asarray(image_homo)

    elif len(images.shape) == 2:
        h, w = images.shape
        image_homo = cv2.warpPerspective(images, M, (h, w))
        return image_homo
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')

def hot_pixel_detection(images, threshold=7):
    """
    Detect deviant pixels inside a image.
    :param image: Image to process.
    :param threshold:
    :returns: Map of deviant pixels
    :rtype:

    """

    if len(images.shape) == 3:
        hot_map = np.ndarray((images.shape[0], images.shape[1], images.shape[2]), np.short)
        for i in range(0, images.shape[0]):
            blur = median_filter(images[i], size=3)
            temp = images[i].astype(np.int) - blur
            im_std = np.std(temp)
            hot_map[i, temp < threshold * im_std] = 0
            hot_map[i, temp >= threshold * im_std] = 1
        return hot_map

    elif len(images.shape) == 2:
        blur = median_filter(images, size=3)
        hot_map = images.astype(np.int) - blur
        im_std = np.std(hot_map)
        hot_map[hot_map < threshold * im_std] = 0
        hot_map[hot_map >= threshold * im_std] = 1
        return hot_map
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def hot_pixel_remove(hot_map, images):
    """
    Equalize hots pixels
    :param hot_map: 
    :param images:
    :return: 
    """
    if len(images.shape) == 3:
        image_whotp = np.array(images).astype(np.uint16)
        for i in range(0, images.shape[0]):
            pixels = np.argwhere(hot_map[i, 1:-2, 1:-2] > 0)
            pixels = pixels + 1
            for pix in pixels:
                image_whotp[i, pix[0], pix[1]] = (images[i, pix[0] + 2, pix[1]].astype(np.uint32) +
                                                  images[i, pix[0] - 2, pix[1]].astype(np.uint32) +
                                                  images[i, pix[0], pix[1] + 2].astype(np.uint32) +
                                                  images[i, pix[0], pix[1] - 2].astype(np.uint32)) / 4
        return image_whotp

    elif len(images.shape) == 2:
        pixels = np.argwhere(hot_map[1:-2, 1:-2] > 0)
        image_whotp = np.array(images)
        pixels = pixels + 1
        for pix in pixels:
            image_whotp[pix[0], pix[1]] = (images[pix[0] + 2, pix[1]].astype(np.uint32) + images[pix[0] - 2, pix[1]].astype(np.uint32) +
                                           images[pix[0], pix[1] + 2].astype(np.uint32) + images[pix[0], pix[1] - 2].astype(np.uint32)) / 4
        return image_whotp
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def roi_image(images, roi):
    """
    Select ROI inside a image.
    :param images: Image to process.
    :param roi: ROI info 2x2 numpy array [[x1,y1][x2,y2]].
    :returns: ROI image.
    :rtype:
    """
    if len(images.shape) == 3:
        return images[:, roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]]
    elif len(images.shape) == 2:
        return images[roi[0, 1]:roi[1, 1], roi[0, 0]:roi[1, 0]]
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def roi_axis(roi, mag=0.2344, pixel_size=2 * 5.86e-3, center=False):
    """
    Calculate the axis in mm
    :param roi: roi info.
    :param mag: Optical system magnification
    :param pixel_size: Pixel size of camera (with bininng)
    :param center: Center the axis on middle of the roi.
    :return: 
    """
    x_axis = np.linspace(.0, (roi[1, 0] - roi[0, 0])
                         * pixel_size / mag, roi[1, 0] - roi[0, 0])

    y_axis = np.linspace(.0, (roi[1, 1] - roi[0, 1])
                         * pixel_size / mag, roi[1, 1] - roi[0, 1])
    if center:
        x_axis = x_axis - (roi[1, 0] - roi[0, 0]) * pixel_size / (2 * mag)
        y_axis = y_axis - (roi[1, 1] - roi[0, 1]) * pixel_size / (2 * mag)
    return x_axis, y_axis


def rot_images(images, angle):
    """
    Rotate images
    :param images: 
    :param angle: 
    :return: 
    """
    if len(images.shape) == 3:
        for i in range(0, images.shape[0]):
            images[i] = interpolation.rotate(images[i], angle, reshape=False)
        return images
    elif len(images.shape) == 2:
        images = interpolation.rotate(images, angle, reshape=False)
        return images
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def gaussian(x, offset, amplitude, mean, stddev):
    """
    Gaussian function for fitting the beam.
    :param x: 
    :param amplitude: 
    :param mean: 
    :param stddev: 
    :return: 
    """
    return amplitude * np.exp(-0.5 * ((x - mean) / stddev) ** 2) + offset


def gaussian_norm(x, offset, amplitude, mean, stddev):
    """
    Gaussian function for fitting the beam.
    :param x:
    :param amplitude:
    :param mean:
    :param stddev:
    :return:
    """
    return amplitude / np.sqrt(stddev) * np.exp(-(x - mean) ** 2 / (2. * stddev ** 2)) + offset


def double_gaussian(x, offset, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    """
    Gaussian function for fitting the beam.
    :param x: 
    :param amplitude: 
    :param mean: 
    :param stddev: 
    :return: 
    """
    # return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2)) \
    #            + amplitude2 / np.sqrt(stddev2) * np.exp(-(x - mean2) ** 2 / (2. * stddev2 ** 2)) + offset
    return amplitude1 * np.exp(-0.5 * ((x - mean1) / stddev1) ** 2) + \
           amplitude2 * np.exp(-0.5 * ((x - mean2) / stddev2) ** 2) + offset


def double_gaussian_norm(x, offset, amplitude1, mean1, stddev1, amplitude2, mean2, stddev2):
    """
    Gaussian function for fitting the beam.
    :param x:
    :param amplitude:
    :param mean:
    :param stddev:
    :return:
    """
    return amplitude1 / np.sqrt(stddev1) * np.exp(-(x - mean1) ** 2 / (2. * stddev1 ** 2)) \
           + amplitude2 / np.sqrt(stddev2) * np.exp(-(x - mean2) ** 2 / (2. * stddev2 ** 2)) + offset


def gaussian_helper(signal, x_axis):
    """ Function that help to compute the initial parameters for the gaussian fit.

    Args:
        signal (np.ndarray): Signal to fit
        x_axis (np.ndarray): Axis for the fit

    Returns:
        double: Return information about signal.
    """
    min = np.min(signal)
    max = np.max(signal)
    max_pos = x_axis[np.argmax(signal)]
    lmh = np.max(x_axis[signal > np.max(signal - min) / 2 + min]) - np.min(
        x_axis[signal > np.max(signal - min) / 2 + min])
    return min, max - min, max_pos, lmh / 2.355, (max - min) / 5, max_pos, lmh


def profile_stats(images):
    """ Compute statistics on the profile signal.

    Args:
        images (np.ndarray): Input image matrix.

    Raises:
        ne.InputError: Wrong dim exception.

    Returns:
        np.ndarray: Output statistics vector.
    """
    # 2D or 3D data
    if len(images.shape) == 3:
        stats = np.zeros((images.shape[0], 4))
        for i in range(0, images.shape[0]):
            profile = images[i].mean(0)
            stats[i] = np.asarray([np.mean(profile), np.std(profile), np.max(profile), np.min(profile)])
        return stats
    elif len(images.shape) == 2:
        profile = images.mean(0)
        stats = np.asarray([np.mean(profile), np.std(profile), np.max(profile), np.min(profile)])
        return stats
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def profile_fwhm(images, x_axis, axis=0):
    """ Calculate the FWHM of the projection of an image.

    Args:
        images (ndarray): Input image.
        x_axis (ndarray): Axis value.
        axis (int, optional): Projection directions. Defaults to 0.

    Raises:
        ne.InputError: Invalid input data.

    Returns:
        ndarray: FWHM parameters
    """
    # 2D or 3D data
    if len(images.shape) == 3:
        fwhm_info = np.zeros((images.shape[0], 4))
        for i in range(0, images.shape[0]):
            profile = images[i].mean(axis)
            y_min = np.min(profile)
            y_max = np.max(profile)
            i_max = np.argmax(profile)
            dr = y_max - y_min
            x_max = x_axis[i_max]

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
            if i_max + a_i > (x_axis.size - 1):
                up = (x_axis.size() - 1)
            else:
                up = i_max + a_i
            fwhm = x_axis[low:i_max + a_i]
            fwhm_info[i] = np.asarray([fwhm[0], fwhm[-1], fwhm_val, x_max])
        return fwhm_info
    elif len(images.shape) == 2:
        profile = images.mean(axis)
        y_min = np.min(profile)
        y_max = np.max(profile)
        i_max = np.argmax(profile)
        dr = y_max - y_min
        x_max = x_axis[i_max]

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
        if i_max + a_i > (x_axis.size - 1):
            up = (x_axis.size() - 1)
        else:
            up = i_max + a_i
        fwhm = x_axis[low:i_max + a_i]
        fwhm_info = np.asarray([fwhm[0], fwhm[-1], fwhm_val, x_max])
        return fwhm_info
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def fit_beam2(images, x_axis, fun=double_gaussian_norm, axis=0):
    """ Fit the projection of an image.

    Args:
        images (ndarray): Input image.
        x_axis (ndarray): Axis value.
        fun (function, optional): The funtion to fit. Defaults to double_gaussian_norm.
        axis (int, optional): Projection directions. Defaults to 0.

    Raises:
        ne.InputError: Invalid input data.

    Returns:
        ndarray: Fit parameters
    """
    # 2D or 3D data
    if len(images.shape) == 3:
        succes = np.zeros(images.shape[0])
        if fun is double_gaussian_norm:
            popt = np.zeros((images.shape[0], 7))
        else:
            popt = np.zeros((images.shape[0], 4))
        for i in range(0, images.shape[0]):
            profile = images[i].mean(axis)
            if fun is double_gaussian_norm:
                p0 = [np.min(profile), np.max(profile), x_axis[np.argmax(profile)], 3,
                      np.max(profile) / 4, x_axis[np.argmax(profile)], 6]
            else:
                p0 = [np.min(profile), np.max(profile), x_axis[np.argmax(profile)], 3]
            try:
                popt[i], pcov = optimize.curve_fit(fun, x_axis, profile, p0=p0)
            except RuntimeError:
                print('Fit failed !')
                succes[i] = 0
            else:
                succes[i] = 1
            # Order fit data
            if popt.shape[1] > 4:
                if popt[i, 1] < popt[i, 4]:
                    if popt[i, 3] > popt[i, 6]:
                        t1 = np.array(popt[i, 1:4])
                        t2 = np.array(popt[i, 4:7])
                        popt[i, 4:7] = t1
                        popt[i, 1:4] = t2
        return popt, succes
    elif len(images.shape) == 2:
        profile = images.mean(axis)
        if fun is double_gaussian_norm:
            p0 = [np.min(profile), np.max(profile), x_axis[np.argmax(profile)], 3,
                  np.max(profile) / 4, x_axis[np.argmax(profile)], 6]
        else:
            p0 = [np.min(profile), np.max(profile), x_axis[np.argmax(profile)], 3]
        try:
            popt, pcov = optimize.curve_fit(fun, x_axis, profile, p0=p0)
        except RuntimeError:
            print('Fit failed !')
        # Order fit data
        if popt.shape[0] > 4:
            if popt[1] < popt[4]:
                if popt[3] > popt[6]:
                    t1 = np.array(popt[1:4])
                    t2 = np.array(popt[4:7])
                    popt[4:7] = t1
                    popt[1:4] = t2
        return popt

    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')


def fit_beam(profile, x_axis, fun=double_gaussian):
    popt, pcov = optimize.curve_fit(fun, x_axis, profile)
    # Order fit data
    if popt[3] < popt[-1]:
        t1 = np.array(popt[1:4])
        t2 = np.array(popt[4:7])
        popt[4:7] = t1
        popt[1:4] = t2
    return popt, pcov


def fft_filter(fft_image, keep_fraction=0.1):
    """ Apply a low pass filter on a FFT image.

    Args:
        fft_image (ndarray): Input image
        keep_fraction (float, optional): Fraction to remove. Defaults to 0.1.

    Returns:
        ndarray: Cutted FFT image.
    """
    im_fft2 = fft_image.copy()
    r, c = im_fft2.shape
    im_fft2[int(r * keep_fraction):int(r * (1 - keep_fraction))] = 0
    im_fft2[:, int(c * keep_fraction):int(c * (1 - keep_fraction))] = 0
    return im_fft2


def mean_confidence_interval(data, confidence=0.95, sem=False):
    """ Calculate the confidence interval of a mean value of data. 

    Args:
        data (np.ndarray): Input data.
        confidence (float, optional): Confidence level. Defaults to 0.95.
        sem (bool, optional): Use only sem for confidence. Defaults to False.

    Returns:
        float, float: mean, interval
    """
    a = 1.0 * np.array(data)
    n = len(a)
    if sem:
        m, h = np.mean(a), sp.stats.sem(a)
    else:
        m, se = np.mean(a), sp.stats.sem(a)
        h = se * sp.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def stats_fit(images, x_axis, axis=0):
    """ Compute statistics on a serie of images.

    Args:
        images (np.ndarray): Input images.
        x_axis (np.ndarray): Axis of the images
        axis (int, optional): Axis direction. Defaults to 0.

    Raises:
        ne.InputError: _description_

    Returns:
        np.ndarray: Stats results vector
    """
    if len(images.shape) == 3:
        stats = np.zeros((images.shape[0], 2))
        for i in range(0, images.shape[0]):
            profile = images[i].mean(axis)
            wi = (profile - np.min(profile)) / np.max(profile)
            stats[i, 0] = np.sum(np.multiply(x_axis, wi)) / np.sum(wi)
            stats[i, 1] = np.sqrt(np.sum(np.multiply(wi, (x_axis - stats[i, 0]) ** 2)) / np.sum(wi))
        return stats
    elif len(images.shape) == 2:
        profile = images.mean(axis)
        wi = (profile - np.min(profile)) / np.max(profile)
        bpos = np.sum(np.multiply(x_axis, wi)) / np.sum(wi)
        bstd = np.sqrt(np.sum(np.multiply(wi, (x_axis - bpos) ** 2)) / np.sum(wi))
        return bpos, bstd
    else:
        raise ne.InputError('Input image must be a 2D image or 3D array of images !')
