from scipy.signal import spectrogram
import pickle
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, periodogram
import scipy.signal as signal
import glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import random


def augment(chunk_data):
    """
    randomly select an augmentation method to generate augmented data
    TODO: rank them, find our beloved methods:)
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    new_chunk_data = copy.deepcopy(chunk_data)
    function_list = [band_pass_filter, channel_flipping, add_noise, masking, amplitude_scaling, shifting]
    func = random.choice(function_list)
    new_chunk_data = func(new_chunk_data)

    return new_chunk_data


def band_pass_filter(chunk_data):
    """
    delete noise using Butterworth order-3 filter, no big difference between from 1-5
    upper bound (0.1, frequency/2)
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    order = 3
    frequency = 5.2084
    nyq = 0.5 * frequency

    lowcut = random.uniform(0.001, 0.05)
    highcut = random.uniform(0.1, nyq)
    low = lowcut / nyq
    high = highcut / nyq

    sos = signal.butter(order, [low, high], btype='bandpass', output='sos')
    filtered_data = signal.sosfiltfilt(sos, chunk_data.T)

    return filtered_data.T


def channel_flipping(chunk_data):
    """
    Original data: Corresponding sensors from the left side and the right of the head are swapped
    The original order:"AB_I_O", "AB_PHI_O", "AB_I_DO", "AB_PHI_DO", "CD_I_O", "CD_PHI_O", "CD_I_DO", "CD_PHI_DO"

    1. We can randomly flip the channel order
    2. or just left side to right side:
    "CD_I_O", "CD_PHI_O", "CD_I_DO", "CD_PHI_DO", "AB_I_O", "AB_PHI_O", "AB_I_DO", "AB_PHI_DO"
    3. O to O, DO to Do
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    flipped_chunk_data = None
    row_len, col_len = chunk_data.shape

    mode_list = ["switch_side", "randomly_flipping"]
    mode = random.choice(mode_list)

    if mode == "switch_side":
        left_data = chunk_data[:, :int(col_len / 2)]
        right_data = chunk_data[:, int(col_len / 2):]
        flipped_chunk_data = np.concatenate((right_data, left_data), axis=1)

    elif mode == "randomly_flipping":
        flipped_chunk_data = chunk_data[:, np.random.permutation(chunk_data.shape[1])]

    return flipped_chunk_data


def generate_noise(chunk_data):
    """
    To generate a noise for passed in data: Additive White Gaussian Noise, physiological(heartbeat, etc.)
    Additive White Gaussian Noise (AWGN):
    Mean should be zero. Standard deviation is generated from raw fNIRS data).
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    noise = None
    mode_list = ["Gaussian"]
    mode = random.choice(mode_list)

    if mode == "Gaussian":
        mean = 0.0
        std = np.std(chunk_data.T, axis=0)
        for i in range(len(std)):
            if noise is None:
                noise = np.random.normal(mean, std[i], chunk_data.shape[1])
            else:
                noise = np.vstack((noise, np.random.normal(mean, std[i], chunk_data.shape[1])))
    return noise


def add_noise(chunk_data):
    """
    To add noise to data (whole or partially)
    specific noise is from function generate_noise
    TODO: we can choose to add one, or two or three of them, or even none.
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    row_len, col_len = chunk_data.shape
    time_span = np.sort(np.random.randint(low=0, high=row_len, size=2))

    start = time_span[0]
    end = time_span[1]

    noised_data = copy.deepcopy(chunk_data)
    noise = generate_noise(chunk_data.T[:, start:end])

    for i in range(col_len):
        noised_data.T[i][start:end] = np.add(noised_data.T[i][start:end], noise[i])

    return noised_data


def masking(chunk_data):
    """
    To mask whole or part of the data with zeros or noise.
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    row_len, col_len = chunk_data.shape
    start = np.random.randint(0,row_len-1)
    end = (np.random.randint(0,row_len//10-1) + start) % row_len
    masking_value = 0.0
    new_chunk_data = copy.deepcopy(chunk_data)
    new_chunk_data[start:end] = masking_value

    mode_list = ["noise", "zeros"]
    mode = random.choice(mode_list)

    if mode == "noise":
        noise = generate_noise(chunk_data.T[:, start:end])
        for i in range(col_len):
            new_chunk_data.T[i][start:end] = np.add(new_chunk_data.T[i][start:end], noise[i])

    return new_chunk_data


def amplitude_scaling(chunk_data):
    """
    Scale ranges in (0.5, 2)
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    new_chunk_data = copy.deepcopy(chunk_data)
    scale = random.uniform(0.5, 2)
    new_chunk_data = np.dot(new_chunk_data, scale)

    return new_chunk_data


def shifting(chunk_data):
    """
    add a constant number (0, 5*std)
    :param chunk_data: NumPy Matrix(timestamps, features)
    :return: augmented chunk_data: NumPy Matrix(timestamps, features)
    """
    new_chunk_data = copy.deepcopy(chunk_data)
    scale = random.uniform(0.0001, 5)
    std = np.std(chunk_data, axis=0)
    shift = np.dot(std, scale)
    new_chunk_data += shift[np.newaxis, :]

    return new_chunk_data