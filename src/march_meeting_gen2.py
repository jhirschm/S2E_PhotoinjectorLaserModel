#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:23:11 2022

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
import RA_CarbideConsulted as RA
import sympy as sp
import pickle as pickle
import pyssnl_20210819 as pyssnl
import S2E_Class

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
#import h5py
from scipy.interpolate import interp1d
import csv
import pandas as pd

import sys
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/dazzler_src/')
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/regen_src/')
sys.path.append('/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/src/sfg_src/')
import DazzlerClass
import RA_CarbideConsulted as RA
import sympy as sp
import pickle as pickle
import pyssnl_20210819 as pyssnl

def resample_method1(input_domain, target_domain, input_vector):
    try:
        f = interp1d(input_domain, input_vector)
        resampled_vector = f(target_domain)
        return resampled_vector
    except ValueError:
        print("Likely the target wavelenth vector is outside the bounds of the input vector (only interpolation\n")

def convert_to_wavelength(I_freq, phase_freq, freq_vec, wavelength_vector_limits):
    '''
    This function converts the frequency domain signal into a wavelength domain.
    Defaults to a wavelength vector that goes from 200nm below to 200nm above the central
    wavelength. This should be adjusted if width of distribution is very large.'''
    c = 299792458.0
    wavelength_vector = np.linspace(wavelength_vector_limits[0],wavelength_vector_limits[1],num=len(freq_vec))
    I_freq_interp = interp1d(2*np.pi*freq_vec, I_freq)
    I_wavelength = (2*np.pi*c/(wavelength_vector**2))*I_freq_interp(2*np.pi*c/wavelength_vector)
    phase_freq_interp = interp1d(2*np.pi*freq_vec, phase_freq)
    phase_wavelength = phase_freq_interp(2*np.pi*c/wavelength_vector)
    return wavelength_vector, I_wavelength, phase_wavelength

def initialize_amplifier(seed_parameters, pump_parameters, cross_section_file_lists, wavelength_pump=np.linspace(978e-9,982e-9,num=20,endpoint=True), wavelength_seed=np.linspace(1010e-9, 1047e-9, num = 14000, endpoint=True), sample_points = 14000):

    amplifier_parameters = [seed_parameters, pump_parameters]
    wavelength_vec, abs_YbKGW, em_YbKGW  = RA.import_spectra_data_csv(cross_section_file_lists[0])
    wavelength_vec_abs_b, abs_YbKGW_b = RA.import_spectra_data_single_csv(cross_section_file_lists[1])
    wavelength_vec_abs_c, abs_YbKGW_c = RA.import_spectra_data_single_csv(cross_section_file_lists[2])
    wavelength_vec_em_b, em_YbKGW_b = RA.import_spectra_data_single_csv(cross_section_file_lists[3])
    wavelength_vec_em_c, em_YbKGW_c = RA.import_spectra_data_single_csv(cross_section_file_lists[4])

    wavelength_vec_abs_b = wavelength_vec_abs_b*1e-9
    wavelength_vec_abs_c = wavelength_vec_abs_c*1e-9
    wavelength_vec_em_b = wavelength_vec_em_b*1e-9
    wavelength_vec_em_c = wavelength_vec_em_c*1e-9
    abs_YbKGW_b = abs_YbKGW_b*1e-20
    abs_YbKGW_c = abs_YbKGW_c*1e-20
    em_YbKGW_b = em_YbKGW_b*1e-20
    em_YbKGW_c = em_YbKGW_c*1e-20

    if (wavelength_vec_abs_b[-1]<1048e-9):
        wavelength_vec_abs_b = np.append(wavelength_vec_abs_b,[1048e-9])
        abs_YbKGW_b = np.append(abs_YbKGW_b,[abs_YbKGW_b[-1]*1e-1])
    if (wavelength_vec_abs_c[-1]<1048e-9):
        wavelength_vec_abs_c = np.append(wavelength_vec_abs_c,[1048e-9])
        abs_YbKGW_c = np.append(abs_YbKGW_c,[abs_YbKGW_c[-1]*1e-1])
    if (wavelength_vec_em_b[-1]<1048e-9):
        wavelength_vec_em_b = np.append(wavelength_vec_em_b,[1048e-9])
        em_YbKGW_b = np.append(em_YbKGW_b,[em_YbKGW_b[-1]*1e-1])

    if (wavelength_vec_em_c[-1]<1048e-9):
        wavelength_vec_em_c = np.append(wavelength_vec_em_c,[1048e-9])
        em_YbKGW_c = np.append(em_YbKGW_c,[em_YbKGW_c[-1]*1e-1])

    print("assuming input cross sections in /cm^2 so converting to /m^2")
    #Preparing Seed 
    sigmas_seed = np.zeros((wavelength_seed.shape[0],7))
    sigmas_seed[:,0] = wavelength_seed
    #a axis conversion
    sigmas_seed[:,2] = resample_method1(wavelength_vec, wavelength_seed, abs_YbKGW)
    sigmas_seed[:,2]= sigmas_seed[:,2]*1e-4
    sigmas_seed[:,1] = resample_method1(wavelength_vec, wavelength_seed, em_YbKGW)
    sigmas_seed[:,1]= sigmas_seed[:,1]*1e-4
    #b axis conversion
    sigmas_seed[:,4] = resample_method1(wavelength_vec_abs_b, wavelength_seed, abs_YbKGW_b)
    sigmas_seed[:,4]= sigmas_seed[:,4]*1e-4
    sigmas_seed[:,3] = resample_method1(wavelength_vec_em_b, wavelength_seed, em_YbKGW_b)
    sigmas_seed[:,3]= sigmas_seed[:,3]*1e-4
    #c axis conversion
    sigmas_seed[:,6] = resample_method1(wavelength_vec_abs_c, wavelength_seed, abs_YbKGW_c)
    sigmas_seed[:,6]= sigmas_seed[:,6]*1e-4
    sigmas_seed[:,5] = resample_method1(wavelength_vec_em_c, wavelength_seed, em_YbKGW_c)
    sigmas_seed[:,5]= sigmas_seed[:,5]*1e-4


    #Preparing Pump 
    sigmas_pump = np.zeros((wavelength_pump.shape[0],7))
    sigmas_pump[:,0] = wavelength_pump
    #a axis conversion
    sigmas_pump[:,2] = resample_method1(wavelength_vec, wavelength_pump, abs_YbKGW)
    sigmas_pump[:,2] = sigmas_pump[:,2]
    sigmas_pump[:,2]= sigmas_pump[:,2]*1e-4
    sigmas_pump[:,1] = resample_method1(wavelength_vec, wavelength_pump, em_YbKGW)
    sigmas_pump[:,1]= sigmas_pump[:,1]*1e-4
    #b axis conversion
    sigmas_pump[:,4] = resample_method1(wavelength_vec_abs_b, wavelength_pump, abs_YbKGW_b)
    sigmas_pump[:,4]= sigmas_pump[:,4]*1e-4
    sigmas_pump[:,3] = resample_method1(wavelength_vec_em_b, wavelength_pump, em_YbKGW_b)
    sigmas_pump[:,3]= sigmas_pump[:,3]*1e-4
    #c axis conversion
    sigmas_pump[:,6] = resample_method1(wavelength_vec_abs_c, wavelength_pump, abs_YbKGW_c)
    sigmas_pump[:,6]= sigmas_pump[:,6]*1e-4
    sigmas_pump[:,5] = resample_method1(wavelength_vec_em_c, wavelength_pump, em_YbKGW_c)
    sigmas_pump[:,5]= sigmas_pump[:,5]*1e-4

    amplifier_parameters[1]["sigmas_pump"] = sigmas_pump
    amplifier_parameters[1]["delta_lambda_pump"] = amplifier_parameters[1]["sigmas_pump"][2,0]- amplifier_parameters[1]["sigmas_pump"][1,0]
    amplifier_parameters[1]["norm_spectral_amplitude_pump"] = 1/(np.sqrt(2*np.pi)*amplifier_parameters[1]["sigma_gauss_pump"])*np.exp((-(amplifier_parameters[1]["sigmas_pump"][:,0]-amplifier_parameters[1]["lambda_0_pump"])**2)/(2*amplifier_parameters[1]["sigma_gauss_pump"]**2))
    amplifier_parameters[1]["spectral_pump_fluence"] = amplifier_parameters[1]["norm_spectral_amplitude_pump"]*amplifier_parameters[1]["pump_fluence"]*amplifier_parameters[1]["delta_lambda_pump"]
    
    
    amplifier_parameters[0]["sigmas_seed"] = sigmas_seed


    return amplifier_parameters
    
def run_amplifier(amplifier_parameters, freq_vector, wavelength_input, data_input_spectrum, data_input_phase, carbide_phase=[0,0,0,0]):
    #wavelength, intensity, phase = trial.convert_to_wavelength(np.abs(E_field_output_ft)**2, np.unwrap(-1*np.arctan2(np.imag(E_field_output_ft), np.real(E_field_output_ft))),freq_vector,[1010e-9,1047e-9])
    c = 299792458
    carbide_intensity_input = data_input_spectrum
    carbide_input_spectral_seed = 1/(np.sqrt(2*np.pi)*amplifier_parameters[0]["sigma_gauss"])*carbide_intensity_input/np.max(carbide_intensity_input)
    sigmas_seed = amplifier_parameters[0]["sigmas_seed"]
    amplifier_parameters[0]["delta_lambda_seed"] = amplifier_parameters[0]["sigmas_seed"][2,0]-amplifier_parameters[0]["sigmas_seed"][1,0]
    amplifier_parameters[0]["norm_spectral_amplitude"] = carbide_input_spectral_seed
    J_pulse_in =amplifier_parameters[0]["norm_spectral_amplitude"]*amplifier_parameters[0]["F_seed"]*amplifier_parameters[0]["delta_lambda_seed"]
    J_pulse_in_normalized = J_pulse_in/np.max(J_pulse_in)
    amplifier_parameters[0]["J_pulse_in"] = J_pulse_in
    amplifier_parameters[0]["J_pulse_in_normalized"] = J_pulse_in_normalized
    #notch not used here, implement this later properly
    notch = RA.generate_gaussian_notch(amplifier_parameters[0]["sigmas_seed"][:,0], 1024.93939e-9, .0, 3.23076923e-9)
    J_pulse_out_carbide, E_pulse_energy_carbide, p_inv_out_seed_carbide, p_inv_out_pump_carbide, number_passes, saturation_condition = RA.run_simulation(amplifier_parameters[1], amplifier_parameters[0], amplifier_parameters[1]["sigmas_pump"], amplifier_parameters[0]["sigmas_seed"],notch)
    #convert to freq domain
    carbide_freq_intensity = J_pulse_out_carbide[:,number_passes-1]*(np.pi*amplifier_parameters[1]["radius_laser_and_pump_mode"]**2)*amplifier_parameters[0]["sigmas_seed"][:,0]**2/(2*np.pi*c)
    temp_freq_vec = c/sigmas_seed[:,0]
    intensity_freq_direct_alt = np.concatenate((np.array([carbide_freq_intensity[0]/10]),carbide_freq_intensity,np.array([carbide_freq_intensity[-1]/10])))
    temp_freq_vec_alt = np.concatenate((np.array([np.min(freq_vector)]), temp_freq_vec,np.array([np.max(freq_vector)])))
    freq_inten_new=resample_method1(temp_freq_vec_alt, freq_vector,intensity_freq_direct_alt)
    #assuming no negative values from output of carbide
    filt = np.zeros(freq_inten_new.shape[0])
    filt[1500:1700] = np.ones(200)
    #freq_inten_new_shifted = freq_inten_new-np.min(freq_inten_new)
    freq_inten_new_shifted = freq_inten_new*filt
    freq_inten_new_shifted_normalized = freq_inten_new_shifted/np.max(freq_inten_new_shifted)
    #carbide_Efield_freq_domain_output = freq_inten_new_shifted_normalized*np.exp(-1j*np.arctan2(np.imag(dazzler_E_field_output_ft),np.real(dazzler_E_field_output_ft)))
    #carbide_Efield_td_output = np.fft.ifft(carbide_Efield_freq_domain_output)
    #carbide_Efield_td_output = carbide_Efield_td_output*np.exp(-1j*(carbide_phase[0]*dazzler_time_vector+carbide_phase[1]*dazzler_time_vector**2+carbide_phase[2]*dazzler_time_vector**3+carbide_phase[3]*dazzler_time_vector**4))
    
    return J_pulse_out_carbide, carbide_freq_intensity, wavelength_input

def amplifier(freq_vector,field_input_ft):
    seed_file_path = "/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/flintSpectrum05142021.csv"
    cross_section_file_list = ['/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/abs_em_YbKGW.csv','/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/abs_b5.csv','/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/abs_c5.csv', '/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/em_b5.csv', '/Users/jackhirschman/Documents/Stanford/PhD_Project/S2E_PhotoinjectorLaserModel/Data/em_c6.csv']
    #define paremters
    parameters_YbKGW = {}
    parameters_YbKGW["tau_gain"] = 0.6e-3 #0.6ms
    parameters_YbKGW["pump_power"] = 120#40 # W
    parameters_YbKGW["pump_time"] = 1e-3 #1ms
    parameters_YbKGW["radius_laser_and_pump_mode"] = .4e-3#.5e-4 #0.5mm
    parameters_YbKGW["pump_fluence"] = parameters_YbKGW["pump_power"]*parameters_YbKGW["pump_time"]/(np.pi*parameters_YbKGW["radius_laser_and_pump_mode"]**2)
    parameters_YbKGW["fwhm_gauss_pump"] = .05e-9#7.1e-9#7.1e-9 # 7.1 nm for 100 KHz rep rate
    parameters_YbKGW["sigma_gauss_pump"] = parameters_YbKGW["fwhm_gauss_pump"]/2.35
    parameters_YbKGW["lambda_0_pump"] = 981.2e-9
    parameters_YbKGW["length_crystal"] = np.sqrt(6.86951515e23)
    parameters_YbKGW["N_gain_ion_density"] = np.sqrt(6.86951515e23)
    parameters_YbKGW["h"] = 6.62606957e-34 #Ws
    parameters_YbKGW["c"] = 3e8 #m/s
    
    parameters_YbKGW["T_losses"] = 0.9
    parameters_YbKGW["N_pump_slices"] = 20
    parameters_YbKGW["pump_inv_start"] = .245 #initial pump inversion 24.5% (vaue for Ho:YLF when transparent)
    #parameters_YbKGW["sigmas_pump"] = sigmas_pump
    
    parameters_YbKGW["crysyal_mixing_ratios"] = [5,0,2.27272727]
    parameters_YbKGW["crysyal_mixing_ratios"] = RA.normalize_mixing(parameters_YbKGW["crysyal_mixing_ratios"])
    
    #Parameters for input pulse
    parameters_seed = {}
    parameters_seed["seed_pulse_duration"] = 1e-12 #used for inversion correction
    parameters_seed["number_single_passes"] = 22
    parameters_seed["N_seed_slices"] = 20
    parameters_seed["radius_laser_and_pump_mode"] = parameters_YbKGW["radius_laser_and_pump_mode"]
    parameters_seed["seed_energy"] = 50e-9#12e-9#based on minimum from flint 4.012e-9 #12 nJ
    parameters_seed["F_seed"] = parameters_seed["seed_energy"]/(np.pi*(parameters_seed["radius_laser_and_pump_mode"])**2)
    parameters_seed["fwhm_gauss"] = 16.8e-9#93.85e-15 (fs) #93.85e-15#10e-9 # shrink temp 20e-9 #20nm
    parameters_seed["sigma_gauss"] = parameters_seed["fwhm_gauss"]/2.35
    parameters_seed["lambda_0"] = 1035e-9 # 1023 nm
    #parameters_seed["delta_lambda_seed"] = parameters_seed["sigmas_seed"][2,0]-parameters_seed["sigmas_seed"][1,0]
    
    sample_points = 100000#16384#14000
    wavelength_pump=np.linspace(978e-9,982e-9,num=20,endpoint=True)
    wavelength_seed=np.linspace(1010e-9, 1047e-9, num = sample_points, endpoint=True) 
    
    amplifier_parameters = initialize_amplifier(parameters_seed,parameters_YbKGW,cross_section_file_list, wavelength_pump=wavelength_pump, wavelength_seed=wavelength_seed, sample_points=sample_points)
    wavelength, spectrum, phase = convert_to_wavelength(np.abs(field_input_ft)**2, np.unwrap(-1*np.arctan2(np.imag(field_input_ft), np.real(field_input_ft))),np.fft.fftshift(freq_vector),[1010e-9,1047e-9])

    return run_amplifier(amplifier_parameters,freq_vector, wavelength, spectrum, phase)

def get_phase(field):
    return np.arctan2(np.imag(field),np.real(field))

def inten_phase_plot(domain,field, xlabel="time (s)",y1label="Norm. Intensity",normalize=True,xlims=None):
    fig,ax = plt.subplots()
    if (normalize):
        factor = np.max(np.abs(field)**2)
    else:
        factor = 1
    ax.plot(domain,np.abs(field)**2/factor,color="red")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(y1label,color = "red", fontsize=14)
    
    ax2=ax.twinx()
    ax2.plot(domain,np.unwrap(get_phase(field)),color="blue")
    ax2.set_ylabel("Phase (Arb.)",color="blue",fontsize=14)
    if (xlims==None):
        xlimits = [domain[0],domain[-1]]
    else:
        xlimits = [xlims[0],xlims[1]]
    plt.xlim(xlimits[0],xlimits[1])
    plt.show()
    
def spec_phase_plot(domain,field, xlabel="frequency (Hz)",y1label="Norm. Intensity",normalize=True,xlims=None,shift_domain=True):
    fig,ax = plt.subplots()
    if(shift_domain):
        shifted_domain=np.fft.fftshift(domain)
    else:
        shifted_domain = domain
    if (normalize):
        factor = np.max(np.abs(field)**2)
    else:
        factor = 1
    ax.plot(shifted_domain,np.fft.fftshift(np.abs(field)**2)/factor,color="red")
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(y1label,color = "red", fontsize=14)
    
    ax2=ax.twinx()
    ax2.plot(shifted_domain,np.unwrap((get_phase(field))),color="blue")
    ax2.set_ylabel("Phase (Arb.)",color="blue",fontsize=14) 
     
    if (xlims==None):
        xlimits = [shifted_domain[0],shifted_domain[-1]]
    else:
        xlimits = [xlims[0],xlims[1]]
    plt.xlim(xlimits[0],xlimits[1])
    plt.show()
    
    
    
central_wavelength = 1024e-9#1035.5e-9#1024e-9
#time_vector = np.linspace(-8.191e-11,8.192e-11,num=16384)
time_vector = np.linspace(-8.191e-11,8.192e-11,num=int(2**15))

#time_vector = np.linspace(-.8e-11,.8e-11,num=40000)

freq_vector = np.fft.fftfreq(n=time_vector.shape[0], d = (time_vector[1]-time_vector[0]))
input_eField = np.sqrt(4.7*10**15)*np.exp(-1.386*((time_vector-0)/(246*10**(-15)))**2)
input_eField_ft = np.fft.fft(input_eField)

omega_dif = 2*np.pi*freq_vector-0#299792458/(central_wavelength)*2*np.pi
delay = 0
sec_order = 0*(1e-15)**2
third_order = 0*-10000*(1e-15)**3
fourth_order = 00*(1e-15)**4
phase = -(delay*omega_dif + sec_order/2 * omega_dif**2 + third_order/6 * omega_dif**3 + 
                 fourth_order/24 * omega_dif**4)


input_eField_ft_alt = input_eField_ft*np.exp(1j*phase)
input_eField = np.fft.ifft(input_eField_ft_alt)



#tests 
'''
field_test = np.exp(-1.386*(time_vector/(246*10**(-15)))**2)
field_test_ft = np.fft.fft(field_test)
field_test2 = np.sqrt(4.7*10**15)*np.exp(-1.386*((time_vector-.5e-11)/(246*10**(-15)))**2)*np.exp(1j*299792458/(central_wavelength)*.5e-11)
field_test2_ft = np.fft.fft(field_test2)

inten_phase_plot(time_vector, field_test,xlims=[-.5e-11,.5e-11])
spec_phase_plot(freq_vector, field_test_ft, y1label="spectrum")
inten_phase_plot(time_vector, field_test2)
spec_phase_plot(freq_vector, field_test2_ft, y1label="spectrum")


'''
#inten_phase_plot(time_vector*1e12, input_eField,xlabel="time (ps)",xlims=[-.5,.5])
#spec_phase_plot(freq_vector, input_eField_ft_alt, y1label="spectrum",xlims=[-5e14,-2e14])

sfg_input = {"E_field":input_eField,"time_vector":time_vector, "frequency_vector":freq_vector,'central_frequency': 299792458/(1024e-9)}
u = pyssnl.UNITS()
ssnl_Obj1 = pyssnl.SSNL(sfg_input)

#self.ssnl_Obj1 = pyssnl.SSNL(self.sfg_input)
#self.ssnl_Obj1.set_default(specphase_2nd_order = 2.561, specphase_3rd_order=.4)
ssnl_Obj1.set_default()
ssnl_Obj1.genEqns()
ssnl_Obj1.genGrids()
ssnl_Obj1.genFields(threshold=1e-5)
ssnl_Obj1.propagate()
#ssnl_Obj1.saveFile(file_name)

#TIME DOMAIN
t_long_dazzler = np.array(sfg_input['time_vector'])/u.ps
t_short_dazzler = ssnl_Obj1.lists['t']/u.ps

#FREQUENCY DOMAIN
f_long_dazzler = np.array(sfg_input['frequency_vector'])
f_short_dazzler = ssnl_Obj1.lists['dOmega']/(2*np.pi)

dazzler_out_fd =  np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.sqrt(np.abs(np.array(sfg_input['E_field']))**2))))

sfg_out_time = ssnl_Obj1.eField['time'][3]

#inten_phase_plot(t_short_dazzler,sfg_out_time[-1],xlims=[-30,30],xlabel="time (ps)")
#spec_phase_plot(f_short_dazzler, np.fft.ifft(sfg_out_time[-1]), y1label="spectrum",shift_domain=False, xlims=[-.2e13,.2e13])


#plt.plot(f_short_dazzler,np.abs(np.fft.fftshift(np.fft.ifft(sfg_out_time[-1])))**2)
#plt.show()


'''
carbide_out, carbide_freq_out, wavelength = amplifier(freq_vector,input_eField_ft)
plt.figure()
plt.plot(wavelength,carbide_out[:,20],color="red")
plt.xlabel("wavelength (m)", fontsize=14)
plt.ylabel("Spectrum",color = "red", fontsize=14)
 ''' 

print("end)")


sods = [0, 5, 0 ,0]
tods = [0, 0, 10000, -10000]

inputs_list = []
inputs_freq_list = []
outputs_list = []
outputs_freq_list = []

for i in range(4):
    input_eField = np.sqrt(4.7*10**15)*np.exp(-1.386*((time_vector-0)/(246*10**(-15)))**2)*np.exp(-1j*299792458/(central_wavelength)*2*np.pi*time_vector)
    input_eField_ft = np.fft.fft(input_eField)

    omega_dif = 2*np.pi*freq_vector-299792458/(central_wavelength)*2*np.pi
    delay = 0
    sec_order = sods[i]*(1e-15)**2
    third_order = tods[i]*(1e-15)**3
    fourth_order = 00*(1e-15)**4
    phase = -(delay*omega_dif + sec_order/2 * omega_dif**2 + third_order/6 * omega_dif**3 + 
                     fourth_order/24 * omega_dif**4)
    
    
    input_eField_ft_alt = input_eField_ft*np.exp(1j*phase)
    input_eField = np.fft.ifft(input_eField_ft_alt)
    
    
    sfg_input = {"E_field":input_eField,"time_vector":time_vector, "frequency_vector":freq_vector,'central_frequency': 299792458/(1024e-9)}
    u = pyssnl.UNITS()
    ssnl_Obj1 = pyssnl.SSNL(sfg_input)
    
    #self.ssnl_Obj1 = pyssnl.SSNL(self.sfg_input)
    #self.ssnl_Obj1.set_default(specphase_2nd_order = 2.561, specphase_3rd_order=.4)
    ssnl_Obj1.set_default()
    ssnl_Obj1.genEqns()
    ssnl_Obj1.genGrids()
    ssnl_Obj1.genFields(threshold=1e-5)
    ssnl_Obj1.propagate()
    #ssnl_Obj1.saveFile(file_name)
    
    #TIME DOMAIN
    t_long_dazzler = np.array(sfg_input['time_vector'])/u.ps
    t_short_dazzler = ssnl_Obj1.lists['t']/u.ps
    
    #FREQUENCY DOMAIN
    f_long_dazzler = np.array(sfg_input['frequency_vector'])
    f_short_dazzler = ssnl_Obj1.lists['dOmega']/(2*np.pi)
    
    dazzler_out_fd =  np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.sqrt(np.abs(np.array(sfg_input['E_field']))**2))))
    
    sfg_out_time = ssnl_Obj1.eField['time'][3]
    
    
    inputs_list.append(input_eField)
    inputs_freq_list.append(input_eField_ft_alt)
    outputs_freq_list.append(np.fft.ifft(sfg_out_time[-1]))
    outputs_list.append(sfg_out_time[-1])
    

fig, axs = plt.subplots(2,4)
for i in range(4):
    field = inputs_list[i]
    domain = time_vector*1e12
    xlabel="time (ps)"
    if (i<2):
        xlims=[-1,1]
    elif (i == 2):
        xlims=[66.7,68.7]
    else:
        xlims=[-68.7,-66.7]
    factor = np.max(np.abs(field)**2)
    axs[0,i].plot(domain,np.abs(field)**2/factor,color="red")
    axs[0,i].set_xlabel(xlabel, fontsize=14)
    if (i==0):
        axs[0,i].set_ylabel("Norm. Intensity",color = "red", fontsize=14)
    
    axs2=axs[0,i].twinx()
    axs2.plot(domain,np.unwrap(get_phase(field)),color="blue")
    if(i==3):
        axs2.set_ylabel("Phase (rad)",color="blue",fontsize=14)
    if (xlims==None):
        xlimits = [domain[0],domain[-1]]
    else:
        xlimits = [xlims[0],xlims[1]]
    plt.xlim(xlimits[0],xlimits[1])
    
    field=inputs_freq_list[i]
    domain = freq_vector
    shift_domain = True
    if(shift_domain):
        shifted_domain=np.fft.fftshift(domain)
    else:
        shifted_domain = domain
    factor = np.max(np.abs(field)**2)
    shifted_domain = 1e-12*shifted_domain

    axs[1,i].plot(shifted_domain,(np.fft.fftshift(np.abs(field)**2))/factor,color="red")
    axs[1,i].set_xlabel("frequency (THz)", fontsize=14)
    if (i==0):
        axs[1,i].set_ylabel("Norm. Spectrum",color = "red", fontsize=14)
    
    axs2=axs[1,i].twinx()
    axs2.plot(shifted_domain,np.unwrap((get_phase(field))),color="blue")
    if (i==3):
        axs2.set_ylabel("Phase (rad)",color="blue",fontsize=14) 
    plt.xlim(-320,-280)
     
plt.subplots_adjust(left=0.037,bottom=0.064,right=0.932,top=0.967,wspace=0.32,hspace=0.154)
plt.show()

fig, axs = plt.subplots(2,4)
for i in range(4):
    field = outputs_list[i]
    domain = t_short_dazzler
    xlabel="time (ps)"
    if (i<2):
        xlims=[-10,10]
    elif (i == 2):
        xlims=[58,78]
    else:
        xlims=[-78,-58]
    factor = np.max(np.abs(field)**2)
    axs[0,i].plot(domain,np.abs(field)**2/factor,color="red")
    axs[0,i].set_xlabel(xlabel, fontsize=14)
    if (i==0):
        axs[0,i].set_ylabel("Norm. Intensity",color = "red", fontsize=14)
    
    axs2=axs[0,i].twinx()
    axs2.plot(domain,np.unwrap(get_phase(field)),color="blue")
    if(i==3):
        axs2.set_ylabel("Phase (Arb.)",color="blue",fontsize=14)
    if (xlims==None):
        xlimits = [domain[0],domain[-1]]
    else:
        xlimits = [xlims[0],xlims[1]]
    #plt.xlim(xlimits[0],xlimits[1])
    
    field=outputs_freq_list[i]
    domain = f_short_dazzler
    shift_domain = False
    if(shift_domain):
        shifted_domain=np.fft.fftshift(domain)
    else:
        shifted_domain = domain
    factor = np.max(np.abs(field)**2)
    shifted_domain = 1e-12*shifted_domain

    axs[1,i].plot(shifted_domain,np.fft.fftshift(np.abs(field)**2)/factor,color="red")
    axs[1,i].set_xlabel("frequency (THz)", fontsize=14)
    if (i==0):
        axs[1,i].set_ylabel("Norm. Spectrum",color = "red", fontsize=14)
    
    axs2=axs[1,i].twinx()
    axs2.plot(shifted_domain,np.unwrap((get_phase(field))),color="blue")
    if (i==3):
        axs2.set_ylabel("Phase (Arb.)",color="blue",fontsize=14) 
    #plt.xlim(-2,2)
     
plt.subplots_adjust(left=0.037,bottom=0.064,right=0.932,top=0.967,wspace=0.32,hspace=0.154)
plt.show()
    
    









