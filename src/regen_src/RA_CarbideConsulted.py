from scipy.integrate import quad
from numpy import genfromtxt
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import re
import math
import scipy.fftpack
import pylab
import h5py
from scipy.interpolate import interp1d
import csv
import pandas as pd

def normalize_mixing(inp):
    a = inp[0]
    b = inp[1]
    c = inp[2]
    total = a + b + c
    return [a/total, b/total, c/total]

def import_spectra_data_csv(filename):
    data=genfromtxt(filename,delimiter=',')
    wavelength = data[:,0]
    absorption = data[:,2]
    emission = data[:,1]
    return wavelength, absorption, emission
def import_spectra_data_single_csv(filename):
    data=genfromtxt(filename,delimiter=',')
    wavelength = data[:,0]
    data = data[:,1]
    return wavelength, data
def import_spectra_oscillator_data_csv(filename,skip_header):
    data=genfromtxt(filename,delimiter=',', skip_header = skip_header)
    wavelength = data[:,0]
    intensity = data[:,1]
    return data
def import_spectra_amp_data_csv(filename,skip_header):
    data=genfromtxt(filename,delimiter=',', skip_header = skip_header)
    wavelength = data[:,0]
    intensity = data[:,1]
    return data

def generate_gaussian_notch(wavelength_vector, central_wavelength, depth, width):
    gaus = 1-depth*np.exp((-(wavelength_vector-central_wavelength)**2)/(2*(width/2.35)**2))
    return {"profile":gaus, "central_wavelength":central_wavelength}


def sub_function_slice_fluence(Spectral_fluence,N_slices,pulse_duration,p_inv_start,sigmas,tau_gain,h,c,N_gain_ion_density,length_crystal,T_losses,notch):
    Ji=Spectral_fluence/N_slices
    dt_slice=pulse_duration/N_slices
    p = np.zeros((N_slices))
    Ji1 = np.zeros((Ji.shape[0],N_slices))
    p[0] = p_inv_start

    for i in range(N_slices-1):
        p[i+1],Ji1[:,i] = sub_func_single_fluence_propagation(p[i], Ji, sigmas, dt_slice, tau_gain, h,c, N_gain_ion_density, length_crystal, T_losses)
        Ji1[:,i] = Ji1[:,i]*notch
    return p[-1], np.sum(Ji1,axis=1)

def sub_func_single_fluence_propagation(p_inv_start,J_pulse_in,sigmas,dt_slice,tau_gain,h,c,N_gain_ion_density,length_crystal,T_losses):
    wavelength = sigmas[:,0]

    p_0 = p_inv_start
    J_sat = h*c/(sigmas[:,0]*(sigmas[:,1]+sigmas[:,2]))
    sigma_g = (p_0*(sigmas[:,1]+sigmas[:,2])-sigmas[:,2])

    Gi = np.exp(sigma_g*N_gain_ion_density*length_crystal)
    Ji = J_sat*T_losses*np.log(1+Gi*(np.exp(J_pulse_in/J_sat)-1))

    spectral_delta_p = (Ji/T_losses-J_pulse_in)*wavelength/(c*h*N_gain_ion_density*length_crystal)
    delta_p = np.sum(spectral_delta_p)

    p_1 = (p_0-delta_p)*np.exp(-dt_slice/tau_gain)

    return p_1, Ji

def run_simulation(pump_parameters,seed_parameters, sigmas_pump, sigmas_seed, notch_settings=None):
    #Assign parameters and inputs for simulation
    J_pulse_in = seed_parameters["J_pulse_in"]

    tau_gain = pump_parameters["tau_gain"]
    h = pump_parameters["h"]
    c = pump_parameters["c"]
    N_gain_ion_density = pump_parameters["N_gain_ion_density"]
    length_crystal = pump_parameters["length_crystal"]
    T_losses = pump_parameters["T_losses"]
    N_pump_slices = pump_parameters["N_pump_slices"]
    spectral_pump_fluence = pump_parameters["spectral_pump_fluence"]
    pump_time = pump_parameters["pump_time"]
    p_inv_start = pump_parameters["pump_inv_start"]
    radius_laser_and_pump_mode=pump_parameters["radius_laser_and_pump_mode"]
    number_single_passes = seed_parameters["number_single_passes"]


    N_seed_slices = seed_parameters["N_seed_slices"]
    seed_pulse_duration = seed_parameters["seed_pulse_duration"]

    p_inv_out_seed = np.zeros((number_single_passes))
    p_inv_out_pump = np.zeros((number_single_passes+1))
    E_pulse_energy = np.zeros((number_single_passes))
    J_spectrum_after_each_single_pass = np.zeros((number_single_passes, J_pulse_in.shape[0]))
    J_pulse_out = np.zeros((J_pulse_in.shape[0],number_single_passes))

    if (notch_settings == None):
        notch_pump = np.ones(spectral_pump_fluence.shape[0])
        notch_seed = np.ones(J_pulse_in.shape[0])
    else:
        if (notch_settings["central_wavelength"]<sigmas_pump[-1,0] and notch_settings["central_wavelength"]>sigmas_pump[0,0]):
            notch_pump = notch_settings["profile"]
            notch_seed = np.ones(J_pulse_in.shape[0])
        elif (notch_settings["central_wavelength"]<sigmas_seed[-1,0] and notch_settings["central_wavelength"]>sigmas_seed[0,0]):
            notch_seed = notch_settings["profile"]
            notch_pump = np.ones(spectral_pump_fluence.shape[0])
        else:
            notch_pump = np.ones(spectral_pump_fluence.shape[0])
            notch_seed = np.ones(J_pulse_in.shape[0])
    mixed_sigmas_pump = np.zeros((sigmas_pump.shape[0],3))
    mixed_sigmas_pump[:,0] = sigmas_pump[:,0]
    mixed_sigmas_pump[:,1] = pump_parameters["crysyal_mixing_ratios"][0]*sigmas_pump[:,1] + pump_parameters["crysyal_mixing_ratios"][1]*sigmas_pump[:,3] + pump_parameters["crysyal_mixing_ratios"][2]*sigmas_pump[:,5]
    mixed_sigmas_pump[:,2] = pump_parameters["crysyal_mixing_ratios"][0]*sigmas_pump[:,2] + pump_parameters["crysyal_mixing_ratios"][1]*sigmas_pump[:,4] + pump_parameters["crysyal_mixing_ratios"][2]*sigmas_pump[:,6]

    mixed_sigmas_seed = np.zeros((sigmas_seed.shape[0],3))
    mixed_sigmas_seed[:,0] = sigmas_seed[:,0]
    mixed_sigmas_seed[:,1] = pump_parameters["crysyal_mixing_ratios"][0]*sigmas_seed[:,1] + pump_parameters["crysyal_mixing_ratios"][1]*sigmas_seed[:,3] + pump_parameters["crysyal_mixing_ratios"][2]*sigmas_seed[:,5]
    mixed_sigmas_seed[:,2] = pump_parameters["crysyal_mixing_ratios"][0]*sigmas_seed[:,2] + pump_parameters["crysyal_mixing_ratios"][1]*sigmas_seed[:,4] + pump_parameters["crysyal_mixing_ratios"][2]*sigmas_seed[:,6]

    p_inv_out_pump[0],J_pulse_out_pump = sub_function_slice_fluence(spectral_pump_fluence,N_pump_slices, pump_time, p_inv_start, mixed_sigmas_pump, tau_gain, h,c,N_gain_ion_density,length_crystal, T_losses, notch_pump)


    number_passes = 0
    saturation_condition = False
    stop_iterating = False
    for i in range(number_single_passes):
        i = number_passes
        p_inv_out_seed[i], J_pulse_out[:,i] = sub_function_slice_fluence(J_pulse_in,N_seed_slices, seed_pulse_duration, p_inv_out_pump[i], mixed_sigmas_seed, tau_gain, h,c,N_gain_ion_density,length_crystal, T_losses, notch_seed)
        J_pulse_in = J_pulse_out[:,i]
        E_pulse_energy[i] = np.sum(J_pulse_out[:,i],0)*(radius_laser_and_pump_mode**2*np.pi)
        p_inv_out_pump[i+1] = p_inv_out_seed[i]

        J_spectrum_after_each_single_pass[i,:]=J_pulse_in/np.max(J_pulse_in)
        #saturation condition when have increase in energy for several steps followed by a decrease
        if (i > 10 and E_pulse_energy[i] <= E_pulse_energy[i-1] and E_pulse_energy[i-1] > E_pulse_energy[i-2] and E_pulse_energy[i-2] > E_pulse_energy[i-3] and E_pulse_energy[i-3] > E_pulse_energy[i-4]):
            saturation_condition = True
        else:
            saturation_condition = False
        #will stop iterating when saturation is reached or when passes max number of iterations
        if (saturation_condition):
            stop_iterating = True
        elif (number_passes >= number_single_passes-1):
            stop_iterating = True
        else:
            stop_iterating = False
            number_passes = number_passes + 1

    return J_pulse_out, E_pulse_energy, p_inv_out_seed, p_inv_out_pump, number_passes, saturation_condition
