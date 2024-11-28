import math
import numpy as np
import tensorflow as tf


def propagate_as(ui, z, wavelength, n0, sampling):
    k = 2 * math.pi / wavelength
    size_pxl = ui.shape
    dfx = 1 / size_pxl[0] / sampling[0]
    dfy = 1 / size_pxl[1] / sampling[1]
    fx = np.fft.fftshift(np.arange(size_pxl[0]) - size_pxl[0] / 2) * dfx
    fy = np.fft.fftshift(np.arange(size_pxl[1]) - size_pxl[1] / 2) * dfy
    fx2 = fx ** 2
    fy2 = fy ** 2

    fy2_2d = np.dot(np.ones([size_pxl[0], 1]), fy2[np.newaxis])
    fx2_2d = np.dot(fx2[:, np.newaxis], np.ones([1, size_pxl[1]]))
    under_sqrt = np.power(n0, 2) - np.power(wavelength, 2) * (fx2_2d + fy2_2d)

    under_sqrt[under_sqrt < 0] = 0
    phase_kernel = k * np.abs(z) * np.sqrt(under_sqrt)

    if z < 0:
        ui = np.conj(ui)

    ftu = np.fft.fft2(ui) * np.exp(1j * phase_kernel)
    uo = np.fft.ifft2(ftu)

    if z < 0:
        uo = np.conj(uo)

    return uo


def propagate_as_tf(ui, z, wavelength, n0, sampling):

        z = tf.cast(z, tf.float64)
        wavelength = tf.cast(wavelength, tf.float64)
        n0 = tf.cast(n0, tf.float64)
        sampling = tf.cast(sampling, tf.float64)
        # Wave number
        k = 2 * math.pi / wavelength

        # Input size
        size_pxl = tf.shape(ui)

        # Frequency increment
        dfx = tf.constant(1.0, dtype=tf.float64) / (tf.cast(size_pxl[0], dtype=tf.float64) * sampling[0])
        dfy = tf.constant(1.0, dtype=tf.float64) / (tf.cast(size_pxl[1], dtype=tf.float64) * sampling[1])

        # Spatial frequencies fx and fy
        fx = tf.signal.fftshift(
            tf.cast(tf.range(size_pxl[0], dtype=tf.float64) - size_pxl[0] / 2, dtype=tf.float64)) * dfx
        fy = tf.signal.fftshift(
            tf.cast(tf.range(size_pxl[1], dtype=tf.float64) - size_pxl[1] / 2, dtype=tf.float64)) * dfy

        # Calculate fx2 and fy2
        fx2 = tf.square(fx)
        fy2 = tf.square(fy)

        # Construct 2D matrices fy2_2d and fx2_2d
        fy2_2d = tf.tensordot(tf.ones([size_pxl[0], 1], dtype=tf.float64), tf.expand_dims(fy2, 0), axes=1)
        fx2_2d = tf.tensordot(tf.expand_dims(fx2, 1), tf.ones([1, size_pxl[1]], dtype=tf.float64), axes=1)

        #  under_sqrt
        under_sqrt = tf.square(n0) - tf.square(wavelength) * (fx2_2d + fy2_2d)

        # Negative value handling of under_sqrt
        under_sqrt = tf.where(under_sqrt < 0, tf.zeros_like(under_sqrt, dtype=tf.float64), under_sqrt)

        # Calculate the phase kernel
        phase_kernel = k * tf.abs(z) * tf.sqrt(under_sqrt)

        # If z < 0, take the complex conjugate
        if z < 0:
            ui = tf.math.conj(ui)

        # Perform FFT and phase adjustment
        ftu = tf.signal.fft2d(ui) * tf.cast(tf.exp(tf.complex(tf.constant(0.0, dtype=tf.float64), phase_kernel)), tf.complex64)

        #  IFFT
        uo = tf.signal.ifft2d(ftu)

        # If z < 0, the result is complex conjugate
        if z < 0:
            uo = tf.math.conj(uo)

        return uo
