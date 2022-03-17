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

#instantiate class instance and define file paths. Next update need to read in parameters from file but for now manually
system = S2E_Class.S2E()
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

sample_points = 16384#14000
wavelength_pump=np.linspace(978e-9,982e-9,num=20,endpoint=True)
wavelength_seed=np.linspace(1010e-9, 1047e-9, num = sample_points, endpoint=True) 

system.initialize_amplifier(parameters_seed,parameters_YbKGW,cross_section_file_list, wavelength_pump=wavelength_pump, wavelength_seed=wavelength_seed, sample_points=sample_points)
system.initialize_initial_seed(seed_file_path)
tester = system.prepare_initial_seed(system.seed_normalized[:,0], system.seed_normalized[:,1])

plt.figure()
plt.plot(system.input_wavelength,system.input)

freq_vector_temp = 3e8/system.input_wavelength
intensity_freq_direct = system.input*system.input_wavelength**2/(2*np.pi*3e8)
max_ind = np.unravel_index(np.argmax(intensity_freq_direct,axis=None), intensity_freq_direct.shape)

plt.figure()
plt.plot(freq_vector_temp, intensity_freq_direct)

freq_space = freq_vector_temp[1]-freq_vector_temp[0]
time_vector = np.fft.fftfreq(freq_vector_temp.shape[0], d=freq_space)
time_vector = np.fft.fftshift(time_vector)
signal = np.fft.fftshift(np.fft.ifft(intensity_freq_direct))*np.exp(1j*2*np.pi*freq_vector_temp[max_ind]*time_vector)
#time = np.arrange(0,)

plt.figure()
plt.plot(time_vector, np.abs(signal)**2)

signal_ft =  np.fft.fftshift(np.fft.fft((signal)))
plt.figure()

plt.plot(np.abs(signal_ft)**2)
plt.show()

#1.2  Set Dazzler parameters
ind = np.argmax(system.seed_normalized[:,1])
central_wavelength = 1035.5e-9#system.seed_normalized[ind,0]
print(system.seed_normalized[ind,0])
width = 93.854716e-15
hole_position = 1022e-9#1024.8e-9
hole_width = 8e-9
hole_depth = 0#.54
delay = 0#5e-13
second_order = 0#300*(1e-15)**2#8*(1e-13)**2#0#1e-26
third_order = 0#16*(1e-14)**3#1e-39
fourth_order =0# 8*(1e-14)**4
dazzler_parameters = {}
dazzler_parameters["central_wavelength"] = central_wavelength
dazzler_parameters["pulse_width"] = width
dazzler_parameters["hole_position"] = hole_position
dazzler_parameters["hole_width"] = hole_width
dazzler_parameters["hole_depth"] = hole_depth
dazzler_parameters["delay"] = delay
dazzler_parameters["second_order"] = second_order
dazzler_parameters["third_order"] = third_order
dazzler_parameters["fourth_order"] = fourth_order

system.initialize_dazzler(dazzler_parameters,selfGenerateInput=False)
time_vector = np.linspace(-2.74e-12,2.74e-12,num=16384)
input_field = np.sqrt(4.7*10**15)*np.exp(-1.386*(time_vector/(246*10**(-15)))**2)*np.exp(-1j*299792458/(central_wavelength)*2*np.pi*time_vector)
system.prepare_initial_seed_time(time_vector, input_field)
system.run_dazzler()
#plt.figure()


#plt.plot(system.dazzler_time_vector,np.unwrap(np.angle(system.dazzler_E_field_input))/np.max(np.unwrap(np.angle(system.dazzler_E_field_input))))
#plt.plot(system.dazzler_time_vector,np.abs(system.dazzler_E_field_input)**2/np.max(np.abs(system.dazzler_E_field_input)**2))
#plt.figure()

def quick_plotter(time_vector, E_fields,labels):
    for E_field in E_fields:
        plt.figure()
        plt.plot(time_vector,np.unwrap(np.angle(E_field))/np.max(np.unwrap(np.angle(E_field))),'b--')
        plt.plot(time_vector,np.abs(E_field)**2/np.max(np.abs(E_field)**2),'b')
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()


quick_plotter(system.dazzler_time_vector, [system.dazzler_E_field_input, system.dazzler_E_field_output],labels=['time (s)','arb'])
#quick_plotter(system.dazzler_freq_vector, [np.fft.fftshift(system.dazzler_E_field_input_ft), np.fft.fftshift(system.dazzler_E_field_output_ft)])
plt.plot(system.dazzler_time_vector,np.unwrap(np.angle(system.dazzler_E_field_input)),'b--')
plt.ylim(0,6000)
#plt.show()
plt.plot(system.dazzler_time_vector,np.unwrap(np.angle(system.dazzler_E_field_output)),'r--')
#plt.ylim(0,6000)
plt.show()
dazzler_out_wavelength, dazzler_out_spectrum, dazzler_out_phase = S2E_Class.convert_to_wavelength(np.abs(system.dazzler_E_field_output_ft)**2, np.unwrap(-1*np.arctan2(np.imag(system.dazzler_E_field_output_ft), np.real(system.dazzler_E_field_output_ft))),system.dazzler_freq_vector,[1010e-9,1047e-9])
plt.figure()
plt.plot(dazzler_out_wavelength,dazzler_out_spectrum/np.max(dazzler_out_spectrum),'b-')
plt.plot(dazzler_out_wavelength,np.unwrap(dazzler_out_phase)/np.max(np.unwrap(dazzler_out_phase)),'b')
plt.xlabel("wavelength (m)")
plt.show()
print("HHEHEHEHEHHEHEHE")
system.run_amplifier(dazzler_out_wavelength,dazzler_out_spectrum,dazzler_out_phase)

system.initialize_sfg()
system.run_sfg("temp")

quick_plotter(system.dazzler_time_vector, [system.carbide_Efield_td_output],labels=['time (s)','arb'])


plt.figure()
#plt.plot(system.t_short_dazzler, abs(system.ssnl_Obj1.eField['time'][3][-1])**2/max(abs(system.ssnl_Obj1.eField['time'][3][-1])**2), label = 'SFG Output', color = 'k')
#plt.xlabel('time (ps)')
#plt.ylabel('Normalized Intensity')
#plt.legend()

plt.plot(system.t_short_dazzler, abs(system.ssnl_Obj1.eField['time'][3][-1])**2, label = 'SFG Output', color = 'k')
plt.xlabel('time (ps)')
plt.ylabel('Normalized Intensity')
plt.legend()
#plt.ylim([-.1,2])
plt.xlim([-20,20])


wavelength_vector, I_wavelength, Phase_wavelength = S2E_Class.convert_to_wavelength(abs(system.ssnl_Obj1.eField['freq'][3][-1])**2, np.angle(system.ssnl_Obj1.eField['freq'][3][-1]),system.ssnl_Obj1.lists['omega'][2]/(2*np.pi), system.ssnl_Obj1.lists['lambda'][2])
print(system.ssnl_Obj1.eField['freq'][3].shape)

plt.figure()
plt.plot(wavelength_vector, I_wavelength, label = 'Intensity')
plt.xlabel('wavelength (m)')
plt.ylabel('Intensity')
#plt.xlim([510e-9,515e-9])
plt.legend()

plt.show()