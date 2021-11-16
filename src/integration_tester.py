#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 15:08:34 2021

@author: jackhirschman
"""

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
import sys
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/dazzler_src/')
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/regen_src/')
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/sfg_src/')

import DazzlerClass
import RA_Notch
import sympy as sp
import pickle as pickle
import pyssnl_20210819 as pyssnl

def import_spectra_data_csv(filename,skip_header):
    data=genfromtxt(filename,delimiter=',', skip_header = skip_header)
    wavelength = data[:,0]
    intensity = data[:,1]
    return data

def convert_to_wavelength(I_freq, phase_freq, freq_vec, wavelength_vector = [None]):
            '''
            This function converts the frequency domain signal into a wavelength domain.
            Defaults to a wavelength vector that goes from 200nm below to 200nm above the central
            wavelength. This should be adjusted if width of distribution is very large.'''
            from scipy.interpolate import interp1d
            c = 299792458.0
            if wavelength_vector[0] == None:
                wavelength_vec_limits=[1030e-9-200e-9, 1030e-9+200e-9]
                wavelength_vector = np.linspace(wavelength_vec_limits[0],wavelength_vec_limits[1],num=len(freq_vec))
            #wavelength_vector = np.linspace(wavelength_vec_limits[0],wavelength_vec_limits[1],num=len(freq_vec))
            I_freq_interp = interp1d(2*np.pi*freq_vec, I_freq)
            I_wavelength = (2*np.pi*c/(wavelength_vector**2))*I_freq_interp(2*np.pi*c/wavelength_vector)
            phase_freq_interp = interp1d(2*np.pi*freq_vec, phase_freq)
            phase_wavelength = phase_freq_interp(2*np.pi*c/wavelength_vector)
            return wavelength_vector, I_wavelength, phase_wavelength

def main():

    '''
    (1) First import data needed for initial Flint spectrum and for running regen. Also define all needed parameters
    (2) Then instantiate Dazzler Class with desired parameters
    (3) Modify flint signal using Dazzler
    (4) Feed modified signal (spectrum only) into Regen
    (5) Take output of regen and add constant phase term to dazzler phase output

    '''
    #(1) Regen import segment and flint data skip flint for now!!!
    #1.1 import YbKGW data and set up grids
    #YbKGW
    number_of_sample_points = 14000 #14000
    wavelength_vec, abs_YbKGW, em_YbKGW  = RA_Notch.import_spectra_data_csv('../Data/abs_em_YbKGW.csv')
    wavelength_pump = np.linspace(978e-9, 982e-9, num = 20, endpoint=True)
    wavelength_seed = np.linspace(1010e-9, 1047e-9, num = number_of_sample_points, endpoint=True)

    #Interpolate at selected frequencies
    sigmas_seed = np.zeros((wavelength_seed.shape[0],3))
    sigmas_seed[:,0] = wavelength_seed
    f = interp1d(wavelength_vec, abs_YbKGW)
    sigmas_seed[:,2] = f(wavelength_seed)*1e-4 #for conversion from cm^2 to m^2
    g = interp1d(wavelength_vec, em_YbKGW)
    sigmas_seed[:,1] = g(wavelength_seed)*1e-4 #for conversion from cm^2 to m^2

    sigmas_pump = np.zeros((wavelength_pump.shape[0],3))
    sigmas_pump[:,0] = wavelength_pump
    f = interp1d(wavelength_vec, abs_YbKGW)
    sigmas_pump[:,2] = f(wavelength_pump)*1e-4 #for conversion from cm^2 to m^2
    g = interp1d(wavelength_vec, em_YbKGW)
    sigmas_pump[:,1] = g(wavelength_pump)*1e-4 #for conversion from cm^2 to m^2


    #Resample Flint Spectrum to use as input
    flintData = RA_Notch.import_spectra_oscillator_data_csv("../Data/flintSpectrum05142021.csv", skip_header = 6)

    flint_resampled = np.zeros(sigmas_seed.shape)
    flint_resampled[:,0] = wavelength_seed
    f = interp1d(flintData[:,0]*1e-9, flintData[:,1])
    flint_resampled[:,1] = f(wavelength_seed)

    flintData_normalized = flint_resampled
    flintData_normalized[:,1] = flint_resampled[:,1]/np.max(flint_resampled[:,1])

    #1.2  Set Dazzler parameters
    ind = np.argmax(flintData_normalized[:,1])
    central_wavelength = flintData_normalized[ind,0]
    width = 93.854716e-15
    hole_position = 1024.8e-9
    hole_width = 8e-9
    hole_depth = 0#.54
    delay = 0#5e-12
    second_order = 0#5e-26
    third_order = 0
    fourth_order = 0
    trial = DazzlerClass.Dazzler_Pulse_Shaper(central_wavelength,width,hole_position,hole_width,hole_depth,delay,second_order,third_order,fourth_order)
    time_vector, EofT = trial.make_gaussian_pulse()
    E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict = trial.shape_input_pulse(EofT, time_vector)

    wavelength, intensity, phase = trial.convert_to_wavelength(np.abs(E_field_output_ft)**2, np.unwrap(-1*np.arctan2(np.imag(E_field_output_ft), np.real(E_field_output_ft))),freq_vector,[1010e-9,1047e-9])
    wavelength_r, intensity_r, phase_r = trial.convert_to_wavelength(np.abs(E_field_output_ft)**2, np.unwrap(-1*np.arctan2(np.imag(E_field_output_ft), np.real(E_field_output_ft))),freq_vector)

    #plt.plot(wavelength_r[6300:7500]*1e9,intensity_r[6300:7500],marker='o')
    #plt.show()

    #1.3 set regen parameters
    parameters_YbKGW = {}
    parameters_YbKGW["tau_gain"] = 0.6e-3 #0.6ms
    parameters_YbKGW["pump_power"] = 2560#40 # W
    parameters_YbKGW["pump_time"] = 1e-3 #1ms
    parameters_YbKGW["radius_laser_and_pump_mode"] = .4e-3#.5e-4 #0.5mm
    parameters_YbKGW["pump_fluence"] = parameters_YbKGW["pump_power"]*parameters_YbKGW["pump_time"]/(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2)
    parameters_YbKGW["fwhm_gauss_pump"] = .05e-9#7.1e-9#7.1e-9 # 7.1 nm for 100 KHz rep rate
    parameters_YbKGW["sigma_gauss_pump"] = parameters_YbKGW["fwhm_gauss_pump"]/2.35
    parameters_YbKGW["lambda_0_pump"] = 981.2e-9
    #parameters_YbKGW["length_crystal"] = 6e-3#14e-3 # 9mm
    #parameters_YbKGW["N_gain_ion_density"] = 1.44e26 #1.44 e28 is 7.25 g/cm^3 to atoms/m^3 for Yb but adjusted by two orders of mag since also like that for HoYLF #2.53e22#2.53e28 # m^3 ()
    parameters_YbKGW["length_crystal"] = 1.04761575e-3
    parameters_YbKGW["N_gain_ion_density"] = 1.66810054e27
    parameters_YbKGW["h"] = 6.62606957e-34 #Ws
    parameters_YbKGW["c"] = 3e8 #m/s

    parameters_YbKGW["T_losses"] = 0.9
    parameters_YbKGW["N_pump_slices"] = 20
    parameters_YbKGW["pump_inv_start"] = .245 #initial pump inversion 24.5% (vaue for Ho:YLF when transparent)
    parameters_YbKGW["sigmas_pump"] = sigmas_pump
    parameters_YbKGW["delta_lambda_pump"] = parameters_YbKGW["sigmas_pump"][2,0]-parameters_YbKGW["sigmas_pump"][1,0]
    parameters_YbKGW["norm_spectral_amplitude_pump"] = 1/(np.sqrt(2*np.pi)*parameters_YbKGW["sigma_gauss_pump"])*np.exp((-(parameters_YbKGW["sigmas_pump"][:,0]-parameters_YbKGW["lambda_0_pump"])**2)/(2*parameters_YbKGW["sigma_gauss_pump"]**2))
    parameters_YbKGW["spectral_pump_fluence"] = parameters_YbKGW["norm_spectral_amplitude_pump"]*parameters_YbKGW["pump_fluence"]*parameters_YbKGW["delta_lambda_pump"]

    #Parameters for input pulse
    parameters_seed = {}
    parameters_seed["sigmas_seed"] = sigmas_seed
    parameters_seed["seed_pulse_duration"] = 1e-12 #used for inversion correction
    parameters_seed["max_number_single_passes"] = 100
    parameters_seed["N_seed_slices"] = 20
    parameters_seed["radius_laser_and_pump_mode"] = parameters_YbKGW["radius_laser_and_pump_mode"]
    parameters_seed["seed_energy"] = 50e-9#12e-9#based on minimum from flint 4.012e-9 #12 nJ
    parameters_seed["F_seed"] = parameters_seed["seed_energy"]/(np.pi*(parameters_seed["radius_laser_and_pump_mode"])**2)
    parameters_seed["fwhm_gauss"] = 16.8e-9#93.85e-15 (fs) #93.85e-15#10e-9 # shrink temp 20e-9 #20nm
    parameters_seed["sigma_gauss"] = parameters_seed["fwhm_gauss"]/2.35
    parameters_seed["lambda_0"] = 1035e-9 # 1023 nm
    parameters_seed["delta_lambda_seed"] = parameters_seed["sigmas_seed"][2,0]-parameters_seed["sigmas_seed"][1,0]
   # parameters_seed["norm_spectral_amplitude"] = 1/(np.sqrt(2*np.pi)*parameters_seed["sigma_gauss"])*flintData_normalized[:,1]
    parameters_seed["norm_spectral_amplitude"] = 1/(np.sqrt(2*np.pi)*parameters_seed["sigma_gauss"])*intensity/np.max(intensity)


    parameters_seed["J_pulse_in"] = parameters_seed["norm_spectral_amplitude"]*parameters_seed["F_seed"]*parameters_seed["delta_lambda_seed"]
    parameters_seed["J_pulse_in_normalized"] = parameters_seed["J_pulse_in"]/np.max(parameters_seed["J_pulse_in"])

    #notch = RA_Notch.generate_gaussian_notch(parameters_seed["sigmas_seed"][:,0], 1024.36e-9, .08, 4e-9)
    #notch = RA_Notch.generate_gaussian_notch(parameters_seed["sigmas_seed"][:,0], 1024.3e-9, .12, 3.95e-9) #pretty decent!
    notch = RA_Notch.generate_gaussian_notch(parameters_seed["sigmas_seed"][:,0], 1024.93939e-9, .0, 3.23076923e-9)

    J_pulse_out_carbide, E_pulse_energy_carbide, p_inv_out_seed_carbide, p_inv_out_pump_carbide, number_passes, saturation_condition = RA_Notch.run_simulation(parameters_YbKGW, parameters_seed, sigmas_pump, sigmas_seed,notch)
    """
    plt.plot(E_pulse_energy_carbide[0:number_passes]*1000)
    plt.yscale("log")
    plt.xlabel("number round trips")
    plt.ylabel("Energy (mJ)")
    plt.show()
    plt.plot(sigmas_seed[:,0]*1e9,J_pulse_out_carbide[:,number_passes-1]*(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2)*1e3)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Energy (mJ)")
    plt.show()
    plt.plot(sigmas_seed[:,0]*1e9,J_pulse_out_carbide[:,number_passes-1]/np.max(J_pulse_out_carbide[:,number_passes-1]),label="output")
    plt.plot(sigmas_seed[:,0]*1e9,parameters_seed["J_pulse_in_normalized"],label="input")
    plt.xlabel("wavelength (nm)")
    plt.ylabel("Normalized Fluence")
    plt.legend()
    plt.show()
    ind = np.argmax(J_pulse_out_carbide[:,number_passes-1])
    print(sigmas_seed[ind,0]*1e9,)

    print(saturation_condition)
    """



    carbideData = import_spectra_data_csv("../Data/carbideSpectrum05142021.csv", skip_header = 6)

    carbide_resampled = np.zeros(sigmas_seed.shape)
    carbide_resampled[:,0] = wavelength_seed
    f = interp1d(carbideData[:,0]*1e-9, carbideData[:,1])
    carbide_resampled[:,1] = f(wavelength_seed)

    carbideData_normalized = carbide_resampled
    carbideData_normalized[:,1] = carbide_resampled[:,1]/np.max(carbide_resampled[:,1])
    """
    plt.plot(carbideData_normalized[:,0]*1e9,carbideData_normalized[:,1])
    plt.plot(sigmas_seed[:,0]*1e9,J_pulse_out_carbide[:,number_passes-1]/np.max(J_pulse_out_carbide[:,number_passes-1]),label="output")
    plt.show()
    """

    #carbide_output_resampled = np.zeros(wavelength_r.shape)
    carbide_out_temp = np.zeros(J_pulse_out_carbide.shape[0]+2)
    carbide_out_temp[0] = 0
    carbide_out_temp[-1] = 0
    carbide_out_temp[1:-1] = J_pulse_out_carbide[:,number_passes-1]*(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2)
    wavelength_temp = np.zeros(sigmas_seed[:,0].shape[0]+2)
    wavelength_temp[0] = wavelength_r[0]
    wavelength_temp[-1] = wavelength_r[-1]
    wavelength_temp[1:-1]= sigmas_seed[:,0]
    f = interp1d(wavelength_temp,carbide_out_temp)
    carbide_output_resampled = f(wavelength_r) #for conversion from cm^2 to m^2

    #convert to frequency domain
    #intensity_wavelength_interp = interp1d(sigmas_seed[:,0],J_pulse_out_carbide[:,number_passes-1]*(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2))
    #intensity_freq_interp = intensity_wavelength_interp(2*np.pi*parameters_YbKGW["c"]/freq_vector[0:7000])*(parameters_YbKGW["c"]/freq_vector[0:7000])**2/(2*np.pi*parameters_YbKGW["c"])
    '''
    freq_interp = interp1d(temp_freq_vec,intensity_freq_direct)
    search_for_min = True
    search_for_max = True
    min_index = None
    max_index = None
    for i in range((freq_vector.shape[0]-1)//2):
        if (freq_vector[i]>temp_freq_vector[0] and search_for_min):
            lower_limit =  freq_vector[i]
            search_for_min = False
            min_index = i
        if (freq_vector[i] >temp_freq_vector[-1] and search_for_max):
            upper_limit =freq_vector[i-1]
            search_for_max = False
            max_index = i
    sampling_freq_vector = freq_vector[min_index:max_index]
    '''
    intensity_freq_direct = J_pulse_out_carbide[:,number_passes-1]*(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2)*sigmas_seed[:,0]**2/(2*np.pi*parameters_YbKGW["c"])
    temp_freq_vec = parameters_YbKGW["c"]/sigmas_seed[:,0]
    intensity_freq_direct_alt = np.concatenate((np.array([intensity_freq_direct[0]/10]),intensity_freq_direct,np.array([intensity_freq_direct[-1]/10])))
    temp_freq_vec_alt = np.concatenate((np.array([np.min(freq_vector)]), temp_freq_vec,np.array([np.max(freq_vector)])))
    freq_interp_alt = interp1d(temp_freq_vec_alt,intensity_freq_direct_alt)
    freq_inten_new = freq_interp_alt(freq_vector)

    #plt.plot(freq_vector[1500:1700],np.abs(E_field_output_ft[1500:1700])**2/np.max(np.abs(E_field_output_ft[1500:1700])**2))
    #plt.plot(freq_vector[1500:1700],freq_inten_new[1500:1700]/np.max(freq_inten_new[1500:1700]))
    #plt.show()

    carbide_freq_domain_output = freq_inten_new*np.exp(-1j*np.arctan2(np.imag(E_field_output_ft),np.real(E_field_output_ft)))
    carbide_td_output = np.fft.ifft(carbide_freq_domain_output)


    added_carbide_phase_1 = 0
    added_carbide_phase_2 = 0
    added_carbide_phase_3 = 0
    added_carbide_phase_4= 0
    carbide_td_output_adjusted = carbide_td_output*np.exp(-1j*(added_carbide_phase_2*time_vector**2+added_carbide_phase_3*time_vector**3+added_carbide_phase_4*time_vector**4))





    #sfg_input = {"E_field":carbide_td_output_adjusted,"time_vector":time_vector,"phase_td":np.arctan2(np.imag(carbide_td_output_adjusted),np.real(carbide_td_output_adjusted)), "frequency_vector":freq_vector,"frequency_spectrum":np.abs(carbide_freq_domain_output)**2, "wavelength_vector":wavelength_r}
    sfg_input = {"E_field":carbide_td_output_adjusted/np.max(carbide_td_output_adjusted),"time_vector":time_vector, "frequency_vector":freq_vector,'central_frequency': 299792458/(1024e-9)}


    #call modified version of SFG code
    u = pyssnl.UNITS()
    ssnl_Obj = pyssnl.SSNL(sfg_input) #ssnl_amy.SSNL(input_pk, spec_phase_1, spec_phase_2) ,1.05*-0.1,-0.5*2.2
    #ssnl_Obj.set_default() #the last two inputs are for the 2nd and 3rd order spectral phases
    ssnl_Obj.set_default(specphase_2nd_order = .9, specphase_3rd_order=.29) #the last two inputs are for the 2nd and 3rd order spectral phases
    ssnl_Obj.genEqns()
    ssnl_Obj.genGrids()
    ssnl_Obj.genFields(threshold=1e-5)
    ssnl_Obj.propagate()
    ssnl_Obj.saveFile('desired_file_name_for_saved peak')

    #make plotting function #######################################################################################
    E_field_dazzler_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sfg_input['E_field'])))
    E_field_dazzler_fd_2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sfg_input['E_field'])))

    #xvals
        #TIME DOMAIN
    t_long_dazzler = np.array(sfg_input['time_vector'])/u.ps
    t_short_dazzler = ssnl_Obj.lists['t']/u.ps

        #FREQUENCY DOMAIN
    f_long_dazzler = np.array(sfg_input['frequency_vector'])
    f_short_dazzler = ssnl_Obj.lists['dOmega']/(2*np.pi)

    #comparison_plot(E_field_original_fd,f_short_original, E_field_dazzler_fd ,np.fft.fftshift( f_long_dazzler), 'Frequency', 'Comparing Frequency Space SFG and Dazzler peaks')

    #plt.figure()
    #plt.plot(sfg_input['time_vector'],sfg_input['E_field'], label = 'Dazzler peak input')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Intensity Envelope')
    #plt.legend()

    #plt.figure()
    #plt.plot(f_short_original, E_field_original_fd, label = 'SFG Self-Constructed Peak (E = 17 uJ)')
    #plt.plot(np.fft.fftshift(f_long_dazzler),E_field_dazzler_fd, label = 'Dazzler peak input (E = 17 uJ)')
    #plt.xlabel('Frequency (1/s)')
    #plt.ylabel(' Envelope')
    #plt.legend()


    #plt.figure()
    #plt.plot(f_short_original, E_field_original_fd/max(E_field_original_fd), label = 'SFG Self-Constructed Peak (E = 17 uJ)')
    #plt.plot(np.fft.fftshift(f_long_dazzler),E_field_dazzler_fd/max(E_field_dazzler_fd), label = 'Dazzler peak input (E = 17 uJ)')
    #plt.xlabel('Frequency (1/s)')
    #plt.ylabel(' Envelope')
    #plt.title('Pre-downsampling')
    #plt.legend()

    #shaping

    ##different efields ######################################################

        #INPUT

    #input_original_td = a.before['time_input']
    input_dazzler_td = sfg_input['E_field']

        #FOURIER TRANSFORM TO FREQUENCY SPACE

    #input_original_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(a.before['time_input'])))
    input_dazzler_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.sqrt(np.abs(np.array(sfg_input['E_field']))**2))))

    efield_td = np.sqrt(np.abs(np.array(sfg_input['E_field']))**2)

        #Just different way of finding the above two
    #input_original_fd_2 = a.before['frequency_input']
    #input_dazzler_fd_2 = ssnl_Obj.before['frequency_input']
        #DOWNSAMPLE
    #dazzler_fd_downsample = ssnl_Obj.before['down_sample_fd']

        #SHAPE in FREQUENCY DOMAIN

    dazzler_fd_shaped = ssnl_Obj.eField['freq'][1][0,:]

        #SHAPE IN TIME DOMAIN

    dazzler_td_shaped = ssnl_Obj.eField['time'][1][0,:]

    ## DIFFERENT PHASES #######################################################################

    #input_original_phase_td = np.unwrap(np.angle(input_original_td))
    input_dazzler_phase_td = np.unwrap(np.angle(input_dazzler_td))

    #phase_original_fd = np.unwrap(np.angle(input_original_fd))
    phase_dazzler_fd = np.unwrap(np.angle(input_dazzler_fd))

    #phase_original_fd_2 = np.unwrap(np.angle(input_original_fd_2 ))
    #phase_dazzler_fd_downsample = np.unwrap(np.angle(dazzler_fd_downsample))


    #plt.figure()
    #plt.plot(t_short_dazzler,abs(ssnl_Obj.eField['time'][1][0,:])**2, color = 'b')
    #plt.plot(t_short_dazzler,abs(ssnl_Obj.eField['time'][2][0,:])**2, color = 'r')
    #plt.title('Dazzler:  Intensity of Chirped Inputs E field')
    #plt.xlabel('Time (ps)')
    #plt.ylabel('Intensity')

    normalizing_const = max(max(ssnl_Obj.eField['time'][2][0,:]**2), max(ssnl_Obj.eField['time'][1][0,:])**2)
    normal_d_chirp_1 = abs(ssnl_Obj.eField['time'][1][0,:])**2/normalizing_const


    normal_d_chirp_2 = abs(ssnl_Obj.eField['time'][2][0,:])**2/normalizing_const
    normalizing_const = max(max(ssnl_Obj.eField['time'][2][0,:]**2), max(ssnl_Obj.eField['time'][1][0,:])**2)

    #plt.figure()
    #plt.plot(t_short_dazzler, normal_d_chirp_1 , color = 'b')
    #plt.plot(t_short_dazzler, normal_d_chirp_2 , color = 'r')
    #plt.xlabel('Time (ps)')
    #plt.ylabel('Normalized Intensity')


    #plt.figure()
    #plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][1][0,:]/ max(max(ssnl_Obj.eField['time'][1][0,:]),max(ssnl_Obj.eField['time'][2][0,:] )) , color = 'b')
    #plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][2][0,:]/ max(max(ssnl_Obj.eField['time'][2][0,:]),max(ssnl_Obj.eField['time'][1][0,:])) , color = 'r')
    #plt.xlabel('Time (ps)')
    #plt.ylabel('Normalized Intensity')


    plt.figure()
    #plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][3][-1]/abs(max(abs(ssnl_Obj.eField['time'][3][-1]))), label = 'Normalized Real Amplitude', color = 'k')
    plt.plot(t_short_dazzler, abs(ssnl_Obj.eField['time'][3][-1])**2/max(abs(ssnl_Obj.eField['time'][3][-1])**2), label = 'SFG Output', color = 'k')
    plt.xlabel('Time (ps)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    #plt.ylim([-.1,2])
    plt.xlim([-5,5])

    """
    plt.figure()
    plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][3][-1]/abs(max(abs(ssnl_Obj.eField['time'][3][-1]))), label = 'Normalized Real Amplitude', color = 'k')
    #plt.plot(t_short_dazzler, abs(ssnl_Obj.eField['time'][3][-1])**2/max(abs(ssnl_Obj.eField['time'][3][-1])**2), label = 'SFG Output', color = 'k')
    plt.xlabel('Time (ps)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.ylim([-2,2])

    plt.figure()
    plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][3][-1]/abs(max(abs(ssnl_Obj.eField['time'][3][-1]))), label = 'Normalized Real Amplitude', color = 'k')
    plt.plot(t_short_dazzler, abs(ssnl_Obj.eField['time'][3][-1])**2/max(abs(ssnl_Obj.eField['time'][3][-1])**2), label = 'Normalized Intensity', color = 'k', linestyle='dashed')
    plt.xlabel('Time (ps)')
    plt.ylabel('Normalized Intensity')
    plt.legend()
    plt.ylim([-2,2])


    plt.figure()
    plt.plot(ssnl_Obj.lists['omega'][2], ssnl_Obj.eField['freq'][3][-1]/abs(max(abs(ssnl_Obj.eField['freq'][3][-1]))), label = 'Normalized Real Amplitude', color = 'r')
    plt.plot(ssnl_Obj.lists['omega'][2], abs(ssnl_Obj.eField['freq'][3][-1])**2/max(abs(ssnl_Obj.eField['freq'][3][-1])**2), label = 'Normalized Intensity', color = 'b')
    plt.xlabel('Frequency (1/s)')
    plt.ylabel('Normalized')
    plt.legend()
    plt.ylim([-2,2])
    """



    wavelength_vector, I_wavelength, Phase_wavelength = convert_to_wavelength(abs(ssnl_Obj.eField['freq'][3][-1])**2, np.angle(ssnl_Obj.eField['freq'][3][-1]),ssnl_Obj.lists['omega'][2]/(2*np.pi), ssnl_Obj.lists['lambda'][2])

    plt.figure()
    plt.plot(wavelength_vector, I_wavelength, label = 'Intensity', color = 'b')
    plt.xlabel('Wavelength (m)')
    plt.ylabel('Intensity')
    plt.xlim([510e-9,515e-9])
    plt.legend()



    c = 299792458
    lambda_list = c/(sfg_input['frequency_vector']+0.00001)
    wavelength_vector, I_wavelength, Phase_wavelength = convert_to_wavelength(abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(sfg_input['E_field']))))**2, np.angle(np.fft.fftshift(np.fft.fft(np.fft.fftshift(sfg_input['E_field'])))),np.fft.fftshift(sfg_input['frequency_vector']))

    #plt.figure()
    #plt.plot(wavelength_vector, I_wavelength/max(I_wavelength), label = 'Intensity', color = 'b')
    #plt.xlabel('Wavelength (m)')
    #plt.ylabel('Intensity')
    #plt.legend()

    plt.show()

if __name__ == "__main__":
    main()
