"""
This module provides utility functions for the analysis of NPM images.
"""
from __future__ import print_function

import datetime
import json
import time
import urllib
import urllib.parse
import urllib.request

import cv2
import h5py
import numpy as np
import pandas as pd


def load_rga(filepath, has_header=False):
    """ Load RGA file.

    Args:
        filepath (path or string): The path or the string of the RGA file.
        has_header (bool, optional): Skip the header of the file. Defaults to False.

    Returns:
        numpy array: The output data.
    """
    df = pd.read_csv(filepath, sep=';')

    max_cylce = int(df['Cycle'].max())
    sizec = df['Cycle'].shape[0]
    sizec = int(sizec / max_cylce)

    data = np.zeros((max_cylce, sizec, 3))

    for i in range(1, max_cylce + 1):
        cycles = df.loc[df['Cycle'] == i].replace(',', '.', regex=True)
        data[i - 1, :, :] = cycles.values[:, 2:].astype(np.float)

    return data


def load_images(filepath, print_info=True):
    """
    Load images from AreaDetector HDF5 file.
    :param filepath: string, Filepath of HDF5
    :returns: images, timestamp, s time, ns time
    :rtype: ndarray

    """
    file = h5py.File(filepath, 'r')
    data = file['entry/data/data'][:]
    timestamp = file['entry/instrument/NDAttributes/NDArrayTimeStamp'][:]
    time_s = file['entry/instrument/NDAttributes/NDArrayEpicsTSSec'][:]
    time_ns = file['entry/instrument/NDAttributes/NDArrayEpicsTSnSec'][:]
    file.close()
    if print_info:
        print('File: ' + filepath + ' successfully loaded.')
        print('Images in this run: ' + str(timestamp.size))
        print('Run starts at: ' + epics_time_str(time_s[0] + 631151999))
        print('Run ends at: ' + epics_time_str(time_s[-1] + 631151999))
    return data, timestamp, time_s, time_ns


def load_attribute(filepath, dataname):
    file = h5py.File(filepath, 'r')
    data = file['entry/instrument/NDAttributes/' + dataname][:]
    file.close()
    return data


def epics_time(timestamp):
    """
    Return a datetime objet from an EPICS timestamp.
    :param timestamp: EPICS timestamp
    :returns: Datetime object
    :rtype: datetime

    """
    time = datetime.datetime.fromtimestamp(
        timestamp)
    return time


def epics_time_str(timestamp, date_format='%Y-%m-%d %H:%M:%S'):
    """
    Return a string from EPICS timestamp.
    :param timestamp: EPICS timestamp
    :param date_format: String format
    :returns: Date in string
    :rtype: string

    """
    time_str = datetime.datetime.fromtimestamp(
        timestamp).strftime(date_format)
    return time_str


def epics_time_ns_str(timestamp, time_ns):
    # TODO: Implement + Docstring
    time_str = datetime.datetime.fromtimestamp(
        timestamp).strftime('%Y_%m_%d_%H_%M_%S')
    return time_str


def export_video(filename, dset, fps):
    frame_size = (dset.shape[2], dset.shape[1])
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    for im in dset:
        im = (im/np.max(im))*255
        im = np.array(im, dtype=np.uint8)
        im_out = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        out.write(im_out)
    out.release()


def extract_images_numpy(filepath, filepath_export=''):
    """
    Save HDF5 images to Numpy array file.
    :param filepath: string, Filepath of HDF5.
    :param filepath_export: string, Filepath of Numpy file.
    :returns: None
    :rtype:

    """

    file = h5py.File(filepath, 'r')
    data = file['entry/data/data'][:]
    date = file['entry/instrument/NDAttributes/NDArrayTimeStamp'][:]
    date_ns = file['entry/instrument/NDAttributes/NDArrayEpicsTSnSec'][:]

    for i in range(0, data.shape[0], 1):
        filename_export = time.strftime(
            '%Y_%m_%d_%H_%M_%S', time.localtime(date[i])) + str(date_ns[i]).replace('.', '_')
        np.save(filename_export + filename_export + '.npy', data[i, :, :])

    return None


def extract_images_tiff(filepath, filepath_export=''):
    """
    Save HDF5 images to tiff images.
    :param filepath: string, Filepath of HDF5.
    :param filepath_export: string, Filepath of Numpy file.
    :returns: None
    :rtype:

    """

    file = h5py.File(filepath, 'r')
    data = file['entry/data/data']
    date = file['entry/instrument/NDAttributes/NDArrayTimeStamp']
    date_ns = file['entry/instrument/NDAttributes/NDArrayEpicsTSnSec']

    for i in range(0, data.shape[0], 1):
        filename_export = time.strftime(
            '%Y_%m_%d_%H_%M_%S', time.localtime(date[i])) + str(date_ns[i]).replace('.', '_')
        cv2.imwrite(filename_export + filename_export +
                    '.tiff',  data[i, :, :])

    return None


def iphi_status(timestamp, verbose=False):
    info = dict.fromkeys([['iphi_lhe:TI_ACCT:IS', SCALAR],
                          ['lhe:dipole', SCALAR],
                          'lhe:QD5', 'lhe:QD4', 'lhe:QD3',
                          'lhe:QF5', 'lhe:QF4', 'lhe:QF2',
                          'iphi:lhe:DV4_LectureI', 'iphi:lhe:DV2_LectureI',
                          'iphi:lhe:DV1_LectureI', 'iphi:lhe:DV3_LectureI',
                          'LHE:X_BPM_1', 'LHE: X_BPM_2', 'LHE: X_BPM_5', 'LHE: X_BPM_6',
                          'LHE:Y_BPM_1', 'LHE:Y_BPM_2', 'LHE:Y_BPM_5', 'LHE:Y_BPM_6'])
    for key, value in info.items():
        value = getDataFromArchiver(key)
    return info


# configuration
PREFIX = 'SOURCE_AUTO_PARAM:'
ADRESS_ARCHIVER = "132.166.31.140"
PORT_ARCHIVER = "17668"

# data type
NOT_A_LIST = 0
LIST = 1
# TYPE PV
SCALAR = 0
WAVEFORM = 1
IMAGE_2D = 2
SIZE_MIN_IMAGE2D = 960 * 600


def secondsToStrLocalTime(sec):
    """
    Return string date from epoch time.
    :param sec: epoch time.
    :return: date string.
    """
    timeStucture = time.localtime(sec)
    # ex: 2018-01-30 16:38:10
    return time.strftime("%Y-%m-%d %H:%M:%S", timeStucture)


def getDataFromArchiver(pvName, pvType, dateTimeStart, dateTimeStop, verbose=False):
    """
    Get PV data from the NPM Archiver Appliance.
    :param pvName: Name of the PV to get.
    :param pvType: Type of the PV to get.
    :param dateTimeStart: String start date.
    :param dateTimeStop:
    :param verbose: Print
    :return: time, PV value
    """
    startProgram = time.time()

    if verbose:
        print("\nPV:", pvName)
        print("start request to archiver.. (%f s)" %
              (time.time() - startProgram))

    # elapsed time between start end stop date
    # Parse a string representing a time according to a format.
    dateStartStructure = time.strptime(dateTimeStart, "%Y-%m-%d %H:%M:%S")
    dateStopStructure = time.strptime(dateTimeStop, "%Y-%m-%d %H:%M:%S")
    # check timeline: start date before stop date
    if (dateStartStructure > dateStopStructure):
        exit("FAILED: date start after date stop")
    elapsedTime = time.mktime(dateStopStructure) - \
        time.mktime(dateStartStructure)

    # split date and time
    dateStart, timeStart = dateTimeStart.split(
        ' ')  # split date in yy/mm/day and hh:mm:ss
    dateStop, timeStop = dateTimeStop.split(' ')

    # UTC+2 could be different to localtime()
    is_dst = time.daylight and time.localtime().tm_isdst > 0
    utc_offset = - (time.altzone if is_dst else time.timezone)
    utc_offset = int(utc_offset / 3600)  # secs to hours
    if utc_offset < 0:
        TIME_DIFFERENCE_WITH_UTC = '000-0' + str(utc_offset) + ':00'
    elif utc_offset > 0:
        TIME_DIFFERENCE_WITH_UTC = '000+0' + str(utc_offset) + ':00'
    else:
        TIME_DIFFERENCE_WITH_UTC = '000+00:00'

    # url construction, data in ISO 8601: yyyy-MM-dd"T"HH:mm:ss.SSSZZ
    dateTimeStart = dateStart + "T" + timeStart + "." + TIME_DIFFERENCE_WITH_UTC
    dateTimeStop = dateStop + "T" + timeStop + "." + TIME_DIFFERENCE_WITH_UTC

    # URL request
    if verbose:
        print(urllib.parse.quote_plus(pvName))
    foramt = "json"  # "raw" "json"
    url = "http://" + ADRESS_ARCHIVER + ":" + PORT_ARCHIVER + "/retrieval/data/getData." + foramt + "?pv=" + urllib.parse.quote_plus(
        pvName) + "&from=" + urllib.parse.quote_plus(dateTimeStart) + "&to=" + urllib.parse.quote_plus(dateTimeStop)

    if (verbose):
        print("..end request to archiver (%f s)\n" %
              (time.time() - startProgram))
        print(url)

    # URL answer
    if verbose:
        print("start receiving data from archiver.. (%f s)" %
              (time.time() - startProgram))
    req = urllib.request.urlopen(url)
    if verbose:
        print("..end receiving data from archiver (%f s)\n" %
              (time.time() - startProgram))
        print("start load data.. (%f s)" % (time.time() - startProgram))

    try:
        data = json.load(req)
    except:
        print("ERROR: no data for the period asked or PV not archived")
        return [], []

    if verbose:
        print("..end load data (%f s)\n" % (time.time() - startProgram))
        print("start split data received.. (%f s)" %
              (time.time() - startProgram))

    secs = [x["secs"] for x in data[0]["data"]]
    vals = [x["val"] for x in data[0]["data"]]
    nanos = [x["nanos"] for x in data[0]["data"]]

    if len(secs) == 0:
        exit("WARNING: 0 data received from archiver")

    # add secs and ns
    for i in range(0, len(secs)):
        secs[i] += float(nanos[i]) / pow(10, 9)  # 1ns = 10^-9sec

    if verbose:
        print("..end split data received(%f s)\n" %
              (time.time() - startProgram))
        print("from", dateTimeStart, "to", dateTimeStop)
        print("Elapsed time:", elapsedTime, "sec")
        print("URL:", url)

    if pvType == SCALAR:
        # check type
        if vals is list:
            exit(pvName + "is a list, not a scalar")
        if verbose:
            print("Nb of data available:", len(vals))
        # display
        if verbose:
            for i in range(len(vals)):
                print(i + 1, ":", vals[i], secondsToStrLocalTime(secs[i]))

    elif pvType == WAVEFORM:
        # check type
        if vals is int or vals is float:
            exit(pvName + "is a scalar")
        elif len(vals[0]) >= SIZE_MIN_IMAGE2D:
            exit(pvName + "is 2D-image, not a waveform")

        if verbose:
            print("Nb of waveform(s) available:", len(vals))
            print("Waveform size:", len(vals[0]))
        # display

        if verbose:
            for i in range(len(vals)):
                print(i + 1, ":", vals[i], secondsToStrLocalTime(secs[i]))

    elif pvType == IMAGE_2D:
        # check type
        if vals is int or vals is float:
            exit(pvName + "is a scalar")
        elif len(vals[0]) < SIZE_MIN_IMAGE2D:
            exit(pvName + "is a waform, not a 2D-image")

        if verbose:
            print("start from vector to matrix.. (%f s)" %
                  (time.time() - startProgram))
        nbOfImage = len(vals)  # vals is a list of images
        if verbose:
            print("number of images:", nbOfImage)

        nbOfPixels = len(vals[0])
        if verbose:
            print("number of pixels per image:", nbOfPixels)
        # check resolution
        if nbOfPixels == 2304000:
            x = 1920
            y = 1200
        elif nbOfPixels == 576000:
            x = 960
            y = 600
        elif nbOfPixels == 1260328:  # MANTA
            x = 1226
            y = 1028
        else:
            exit("ERROR: resolution")
        if verbose:
            print("resolution:", x, "x", y)

        # reshape: vector to matrix (2D image)
        for i in range(nbOfImage):
            # 1D array to 2D array (2D-image)
            # vals[i] is a list (1D) which contains 1 image (2D)
            data = np.array(vals[i]).reshape((y, x))
            vals[i] = data
        if verbose:
            print("..end from vector to matrix (%f s)" %
                  (time.time() - startProgram))
    else:
        exit("PV is neither a SCALAR neither a WAVEFORM neither a IMAGE_2D")

    return secs, vals


def TicTocGenerator():
    # Generator that returns time differences
    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator


# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %e seconds.\n" % tempTimeInterval)


def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
