"""
The code implements the forward model for phase retrival from two defocused intensity images.
Your task is to solve the inverse problem,
that is to find the phase1 that will result in intensity2 prediction that is very close to the intensity2.

All distances are given in um
"""

import numpy as np
import scipy.io as sio
from utils import propagate_as
from utils import propagate_as_tf
import matplotlib.pyplot as plt

import tensorflow as tf


if __name__ == '__main__':
    ####################################################################################################################
    # Simulation of the measurement data
    ####################################################################################################################
    s = input("Please choose algorithm,input A or B (Gradient Descent:A,Conjugate descent:B):")

    wavelength = 0.561  # wavelength
    n0 = 1  # Refractive index of air
    sampling = np.array([2.4, 2.4])     # The physical distance between each pixel of the camera, horizontally and vertically
    z_vec = np.array([3.5e3, 11.5e3])   # The distance between the two samples and the camera
    M = 1024
    N = 1024
    
    filename = "optical_field_at_object_plane.mat"
    if not filename:
        raise Exception
    data_obj = sio.loadmat(filename)  # Load this file as a dictionary
    amp_obj = data_obj['amp_obj']  # Extract the magnitude of the object plane from the dictionary
    phase_obj = data_obj['phase_obj']  # Extract the phase of the object plane from the dictionary

    # Light field in the object plane
    u_obj = amp_obj * np.exp(1j*phase_obj)

    # Use the propagation function to calculate the light field of sampling surface 1: u1
    u1 = propagate_as(u_obj, z_vec[0], wavelength, n0, sampling)
    # Use the propagation function to calculate the light field of sampling surface 2: u2
    u2 = propagate_as(u_obj, z_vec[1], wavelength, n0, sampling)

    # The intensity of the light field at sampling surfaces 1 and 2
    i1 = np.abs(u1)
    i2 = np.abs(u2)

    # The true phase information of the light field u1 is extracted
    phase1 = np.angle(u1)

    ####################################################################################################################
    # The actual forward model
    ####################################################################################################################
    # phase1_est = phase1 # CHECK OUT THIS OPTION - UNCOMMENT THIS LINE

    def cost_current_Function(i2_r, i2_t, M_, N_):     # current cost formular

        cost = np.sqrt(np.sum(np.square(i2_r - i2_t)) / (M_ * N_))
        return cost


    def Conjugate_Gradient(phase1, current_phase1_est, i1, i2, M, N, z_vec, wavelength, n0, sampling, max_iter,
                           tol=1e-7):

        # Convert inputs to TensorFlow tensors
        i1_tensor = tf.convert_to_tensor(i1, dtype=tf.complex64)
        i2_tensor = tf.convert_to_tensor(i2, dtype=tf.complex64)
        M = tf.convert_to_tensor(M, dtype=tf.float64)
        N = tf.convert_to_tensor(N, dtype=tf.float64)
        wavelength = tf.convert_to_tensor(wavelength, dtype=tf.complex64)
        n0 = tf.convert_to_tensor(n0, dtype=tf.complex64)
        sampling = tf.convert_to_tensor(sampling, dtype=tf.complex64)
        z_diff = tf.convert_to_tensor(z_vec[1] - z_vec[0], dtype=tf.complex64)

        # Initialize the estimated phase and the residual
        phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)
        r_k = None  # Current residual
        p_k = None  # Current search direction

        for k in range(max_iter):
            with tf.GradientTape() as tape:
                # Calculate the assumed optical field u1
                u1_es = tf.cast(tf.sqrt(i1_tensor), tf.complex64) * tf.exp(
                    tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
                # Use the propagation function to compute the estimated optical field u2
                u2_es = propagate_as_tf(u1_es, z_diff, wavelength, n0, sampling)
                # Calculate the intensity of the estimated optical field u2
                i2_es = tf.square(tf.abs(u2_es))

                # Convert intensity to calculate the loss
                i2_tensor = tf.cast(tf.abs(i2_tensor), tf.float64)
                i2_es = tf.cast(tf.abs(i2_es), tf.float64)

                # Loss function
                cost_ = tf.sqrt(tf.reduce_sum(tf.square(i2_es - i2_tensor)) / (M * N))

            # Compute the gradient
            grad = tape.gradient(cost_, phase1_es)

            # Initialize the residual and search direction
            if k == 0:
                r_k = grad
                p_k = -r_k
            else:
                # Compute the conjugate coefficient
                r_k_new = grad
                beta_k = tf.reduce_sum(tf.square(r_k_new)) / tf.reduce_sum(tf.square(r_k))
                p_k = -r_k_new + beta_k * p_k
                r_k = r_k_new

            # Compute the step size alpha_k
            alpha_k = tf.reduce_sum(tf.square(r_k)) / tf.reduce_sum(tf.square(p_k))

            # Update the phase estimate phase1_es
            phase1_es.assign_add(alpha_k * p_k)

            cost_current_in = cost_current_Function(i2, i2_es, M, N)
            Costs.append(cost_current_in)
            phase_cost_in = np.sqrt(np.sum(np.square(np.float64(phase1_es) - phase1)) / (M * N))
            Cost_phase.append(phase_cost_in)

            # Check for convergence
            if tf.sqrt(tf.reduce_sum(tf.square(tf.abs(grad)))) < tol:
                break

        return phase1_es.numpy(), i2_es.numpy()

        # Set the initial phase value to facilitate gradient descent optimization


    def Gradient_Descent(alpha, current_phase1_est, i1, i2, M, N, z_vec, wavelength, n0, sampling):   # execute once

        i1_tensor = tf.convert_to_tensor(i1, dtype=tf.complex64)                 # tf
        i2_tensor = tf.convert_to_tensor(i2, dtype=tf.complex64)
        alpha = tf.convert_to_tensor(alpha, dtype=tf.complex64)
        M = tf.convert_to_tensor(M, dtype=tf.float64)
        N = tf.convert_to_tensor(N, dtype=tf.float64)
        wavelength = tf.convert_to_tensor(wavelength, dtype=tf.complex64)
        n0 = tf.convert_to_tensor(n0, dtype=tf.complex64)
        sampling = tf.convert_to_tensor(sampling, dtype=tf.complex64)
        z_diff = tf.convert_to_tensor(z_vec[1] - z_vec[0], dtype=tf.complex64)

        phase1_es = tf.Variable(current_phase1_est, dtype=tf.complex64)

        with tf.GradientTape() as tape:
            # Given the intensity i1, construct the hypothetical light field u1
            u1_es = tf.cast(tf.sqrt(i1_tensor), tf.complex64) * tf.exp(tf.complex(0.0, tf.cast(phase1_es, tf.float32)))
            # After the propagation function, the estimated light field u2 is calculated
            u2_es = propagate_as_tf(u1_es, z_diff, wavelength, n0, sampling)
            # Calculate the estimated intensity i2 of the light field u2
            i2_es = tf.square(tf.abs(u2_es))

            i2_tensor = tf.cast((tf.abs(i2_tensor)), tf.float64)
            i2_es = tf.cast((tf.abs(i2_es)), tf.float64)

            # use TensorFlow to get cost_
            cost_ = tf.sqrt(tf.reduce_sum(tf.square(i2_es - i2_tensor)) / (M * N))

        # Calculate the gradient of cost_ with respect to phase1_es,and renew
        dy_dx = tape.gradient(cost_, phase1_es)
        phase1_es.assign_sub(alpha * dy_dx)
        # return new phase1_est
        return phase1_es.numpy(), i2_es.numpy()

        # Set the initial phase value to facilitate gradient descent optimization

    phase1_est = np.zeros_like(phase1)

    if s == "A":
        iters_ = 200
        Costs = []
        Cost_phase = []
        i2_est = np.zeros_like(i2)
        for i in range(iters_):  # execute 200 times Gradient_Descent
            phase1_est, i2_est = Gradient_Descent(572, phase1_est, i1, i2, M, N, z_vec, wavelength, n0, sampling)
            cost_current = cost_current_Function(i2, i2_est, M, N)
            Costs.append(cost_current)

            phase_cost = np.sqrt(np.sum(np.square(np.float64(phase1_est) - phase1)) / (M * N))
            Cost_phase.append(phase_cost)

    else:
        iters_ = 500
        Costs = []
        Cost_phase = []
        i2_est = np.zeros_like(i2)
        phase1_est, i2_est = Conjugate_Gradient(phase1, phase1_est, i1, i2, M, N, z_vec, wavelength, n0, sampling, iters_)

    phase1_est = np.float64(phase1_est)
    print(phase1_est)
    print('Final cost = ', cost_current_Function(i2, i2_est, M, N))

    u_1 = i1 * np.exp(1j * phase1_est)
    u_0 = propagate_as(u_1, -z_vec[0], wavelength, n0, sampling)
    i0 = np.abs(u_0)

    fig, ax = plt.subplots()
    ax.plot(np.arange(iters_), Costs)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    fig0, ax = plt.subplots()
    ax.plot(np.arange(iters_), Cost_phase)
    ax.set(xlabel='Iters', ylabel='Cost_phase', title='Cost_phase along iters')

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 8))
    im1 = ax1.imshow(i2_est ** 2)
    ax1.set_title("Predicted intensity at z2")
    im2 = ax2.imshow(i2 ** 2)
    ax2.set_title("The actual intensity at z2")
    im3 = ax3.imshow(i0 ** 2)
    ax3.set_title("Predicted intensity at z0")
    plt.show()