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

h = 6.62606957e-34
c = 3e8 
nm = 1e-9
ns = 1e-9
ps = 1e-12
fs = 1e-15

class S2E():
    def __init__(self):
        #constants
        h = 6.62606957e-34
        c = 3e8 
        nm = 1e-9
        ns = 1e-9
        ps = 1e-12
        fs = 1e-15
        self.pulse_initialized = False
        self.selfGeneratedInput = False
        self.amplifier_initialized = False


    '''
    Requires wavelength and spectrum for input as well as a time vector which should be
    discretized based on a reptition rate of 240/pulse width.
    Optionally can use input phase for initial pulse. 
    '''
    def prepare_initial_seed(self, wavelength, spectrum, phase_input_td=0):
        self.input = spectrum
        self.input_wavelength = wavelength
        #time_vector = np.li
        input_intensity = convert_wavelength_to_time(self.input_wavelength,self.input)
        #self.electric_field = construct_timeDomain_Efield(time_vector, input_intensity, phase_input_td)
        #self.time_vector = time_vector
        self.pulse_initialized = True
        return input_intensity
    
    def prepare_initial_seed_time(self, time_vector, eField):
        self.electric_field = eField
        self.time_vector = time_vector
        self.pulse_initialized = True
       

    def resample_method1(self, input_domain, target_domain, input_vector):
        try:
            f = interp1d(input_domain, input_vector)
            resampled_vector = f(target_domain)
            return resampled_vector
        except ValueError:
            print("Likely the target wavelenth vector is outside the bounds of the input vector (only interpolation\n")
    def initialize_dazzler(self, dazzler_parameters,selfGenerateInput=False):
        self.central_wavelength = dazzler_parameters["central_wavelength"]
        self.pulse_width = dazzler_parameters["pulse_width"]
        self.hole_position = dazzler_parameters["hole_position"]
        self.hole_width = dazzler_parameters["hole_width"]
        self.hole_depth = dazzler_parameters["hole_depth"]
        self.delay = dazzler_parameters["delay"]
        self.second_order = dazzler_parameters["second_order"]
        self.third_order = dazzler_parameters["third_order"]
        self.fourth_order = dazzler_parameters["fourth_order"]

        self.dazzlerObject = DazzlerClass.Dazzler_Pulse_Shaper(self.central_wavelength,self.pulse_width,self.hole_position,self.hole_width,self.hole_depth,self.delay,self.second_order,self.third_order,self.fourth_order)
        if(selfGenerateInput):
            self.selfGeneratedInput = True
            self.time_vector, self.electric_field = self.dazzlerObject.make_gaussian_pulse()
            print("generating input signal\n")
           

    def update_dazzler_params(self, new_dazzler_parameters):
        self.central_wavelength = new_dazzler_parameters["central_wavelength"]
        self.pulse_width = new_dazzler_parameters["pulse_width"]
        self.hole_position = new_dazzler_parameters["hole_position"]
        self.hole_width = new_dazzler_parameters["hole_width"]
        self.hole_depth = new_dazzler_parameters["hole_depth"]
        self.delay = new_dazzler_parameters["delay"]
        self.second_order = new_dazzler_parameters["second_order"]
        self.third_order = new_dazzler_parameters["third_order"]
        self.fourth_order = new_dazzler_parameters["fourth_order"]


        self.dazzlerObject.position = self.central_wavelength #this is central wavelength in m
        self.dazzlerObject.pulsewidth = self.pulse_width #this is in time (seconds)

        self.dazzlerObject.hole_position = self.hole_position
        self.dazzlerObject.hole_depth = self.hole_depth #0 to 1
        self.dazzlerObject.hole_width = self.hole_width

        self.dazzlerObject.delay = self.delay #0 to 90e-13
        self.dazzlerObject.sec_order = self.sec_order #-60e-26 to 60e-26
        self.dazzlerObject.third_order = self.third_order #80e-39 to 80e-39
        self.dazzlerObject.fourth_order = self.fourth_order #-220e-52 to 220e-52

        self.dazzlerObject.width = 2.18*self.position**2/(self.c*self.pulsewidth)
        self.dazzlerObject.omega0 = 2*np.pi*self.c/self.position
        self.dazzlerObject.chi0 = self.width/(2*self.position)
        self.dazzlerObject.del_omega0 = self.omega0*(self.chi0-self.chi0**3)
        self.dazzlerObject.omega1 = 2*np.pi*self.c/self.hole_position
        self.dazzlerObject.chi1 = self.hole_width/(2*self.hole_position)
        self.dazzlerObject.del_omega1 = self.omega1*(self.chi1-self.chi1**3)/2
    
    
    def initialize_initial_seed(self, file_path):
        print("Make sure to match expected data format. Expecting Flint file.\n")
        seedData = RA.import_spectra_oscillator_data_csv(file_path, skip_header = 6)
       
        seed_resampled = np.zeros(self.sigmas_seed[:,0:2].shape)
        seed_resampled[:,0] = self.sigmas_seed[:,0]
        seed_resampled[:,1] = self.resample_method1(seedData[:,0]*1e-9,self.sigmas_seed[:,0],seedData[:,1])
        self.seed_normalized = seed_resampled
        self.seed_normalized[:,1] = seed_resampled[:,1]/np.max(seed_resampled[:,1])
    
    '''
    Runs the dazzler code
    Requires input pulse in frequency domain
    '''
    def run_dazzler(self):
        if (self.pulse_initialized or self.selfGeneratedInput):
            self.dazzler_E_field_input, self.dazzler_E_field_input_ft, self.dazzler_E_field_output, self.dazzler_E_field_output_ft,self.dazzler_time_vector, self.dazzler_freq_vector, self.dazzler_components_dict = self.dazzlerObject.shape_input_pulse(self.electric_field, self.time_vector, E_field_input_freq_domain=False)
        
        else:
            print("Dazzler does not have a valid input pulse or parameters.\n")

        return 
    '''
    Expects seed and pump parameters as well as list of files for cross sections. For now the file read in is a bit specific. For primary axis, requires abs and em together. Then abs b abs c em b em c. 
    These parameters do not include the input pulse to the carbide like in the original code version. 
    '''
    def initialize_amplifier(self, seed_parameters, pump_parameters, cross_section_file_lists, wavelength_pump=np.linspace(978e-9,982e-9,num=20,endpoint=True), wavelength_seed=np.linspace(1010e-9, 1047e-9, num = 14000, endpoint=True), sample_points = 14000):

        self.amplifier_parameters = [seed_parameters, pump_parameters]
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
        self.sigmas_seed = np.zeros((wavelength_seed.shape[0],7))
        self.sigmas_seed[:,0] = wavelength_seed
        #a axis conversion
        self.sigmas_seed[:,2] = self.resample_method1(wavelength_vec, wavelength_seed, abs_YbKGW)
        self.sigmas_seed[:,2]=self.sigmas_seed[:,2]*1e-4
        self.sigmas_seed[:,1] = self.resample_method1(wavelength_vec, wavelength_seed, em_YbKGW)
        self.sigmas_seed[:,1]=self.sigmas_seed[:,1]*1e-4
        #b axis conversion
        self.sigmas_seed[:,4] = self.resample_method1(wavelength_vec_abs_b, wavelength_seed, abs_YbKGW_b)
        self.sigmas_seed[:,4]=self.sigmas_seed[:,4]*1e-4
        self.sigmas_seed[:,3] = self.resample_method1(wavelength_vec_em_b, wavelength_seed, em_YbKGW_b)
        self.sigmas_seed[:,3]=self.sigmas_seed[:,3]*1e-4
        #c axis conversion
        self.sigmas_seed[:,6] = self.resample_method1(wavelength_vec_abs_c, wavelength_seed, abs_YbKGW_c)
        self.sigmas_seed[:,6]=self.sigmas_seed[:,6]*1e-4
        self.sigmas_seed[:,5] = self.resample_method1(wavelength_vec_em_c, wavelength_seed, em_YbKGW_c)
        self.sigmas_seed[:,5]=self.sigmas_seed[:,5]*1e-4


        #Preparing Pump 
        self.sigmas_pump = np.zeros((wavelength_pump.shape[0],7))
        self.sigmas_pump[:,0] = wavelength_pump
        #a axis conversion
        self.sigmas_pump[:,2] = self.resample_method1(wavelength_vec, wavelength_pump, abs_YbKGW)
        self.sigmas_pump[:,2] =self.sigmas_pump[:,2]
        self.sigmas_pump[:,2]=self.sigmas_pump[:,2]*1e-4
        self.sigmas_pump[:,1] = self.resample_method1(wavelength_vec, wavelength_pump, em_YbKGW)
        self.sigmas_pump[:,1]=self.sigmas_pump[:,1]*1e-4
        #b axis conversion
        self.sigmas_pump[:,4] = self.resample_method1(wavelength_vec_abs_b, wavelength_pump, abs_YbKGW_b)
        self.sigmas_pump[:,4]=self.sigmas_pump[:,4]*1e-4
        self.sigmas_pump[:,3] = self.resample_method1(wavelength_vec_em_b, wavelength_pump, em_YbKGW_b)
        self.sigmas_pump[:,3]=self.sigmas_pump[:,3]*1e-4
        #c axis conversion
        self.sigmas_pump[:,6] = self.resample_method1(wavelength_vec_abs_c, wavelength_pump, abs_YbKGW_c)
        self.sigmas_pump[:,6]=self.sigmas_pump[:,6]*1e-4
        self.sigmas_pump[:,5] = self.resample_method1(wavelength_vec_em_c, wavelength_pump, em_YbKGW_c)
        self.sigmas_pump[:,5]=self.sigmas_pump[:,5]*1e-4

        self.amplifier_parameters[1]["sigmas_pump"] = self.sigmas_pump
        self.amplifier_parameters[1]["delta_lambda_pump"] = self.amplifier_parameters[1]["sigmas_pump"][2,0]-self.amplifier_parameters[1]["sigmas_pump"][1,0]
        self.amplifier_parameters[1]["norm_spectral_amplitude_pump"] = 1/(np.sqrt(2*np.pi)*self.amplifier_parameters[1]["sigma_gauss_pump"])*np.exp((-(self.amplifier_parameters[1]["sigmas_pump"][:,0]-self.amplifier_parameters[1]["lambda_0_pump"])**2)/(2*self.amplifier_parameters[1]["sigma_gauss_pump"]**2))
        self.amplifier_parameters[1]["spectral_pump_fluence"] = self.amplifier_parameters[1]["norm_spectral_amplitude_pump"]*self.amplifier_parameters[1]["pump_fluence"]*self.amplifier_parameters[1]["delta_lambda_pump"]
        
        
        self.amplifier_parameters[0]["sigmas_seed"] = self.sigmas_seed


        self.amplifier_initialized = True
        
    def run_amplifier(self,wavelength_input, data_input_spectrum, data_input_phase, carbide_phase=[0,0,0,0]):
        #wavelength, intensity, phase = trial.convert_to_wavelength(np.abs(E_field_output_ft)**2, np.unwrap(-1*np.arctan2(np.imag(E_field_output_ft), np.real(E_field_output_ft))),freq_vector,[1010e-9,1047e-9])
        self.carbide_intensity_input = data_input_spectrum
        self.carbide_input_spectral_seed = 1/(np.sqrt(2*np.pi)*self.amplifier_parameters[0]["sigma_gauss"])*self.carbide_intensity_input/np.max(self.carbide_intensity_input)
        
        self.amplifier_parameters[0]["delta_lambda_seed"] = self.amplifier_parameters[0]["sigmas_seed"][2,0]-self.amplifier_parameters[0]["sigmas_seed"][1,0]
        self.amplifier_parameters[0]["norm_spectral_amplitude"] = self.carbide_input_spectral_seed
        self.J_pulse_in = self.amplifier_parameters[0]["norm_spectral_amplitude"]*self.amplifier_parameters[0]["F_seed"]*self.amplifier_parameters[0]["delta_lambda_seed"]
        self.J_pulse_in_normalized = self.J_pulse_in/np.max(self.J_pulse_in)
        self.amplifier_parameters[0]["J_pulse_in"] = self.J_pulse_in
        self.amplifier_parameters[0]["J_pulse_in_normalized"] = self.J_pulse_in_normalized
        #notch not used here, implement this later properly
        notch = RA.generate_gaussian_notch(self.amplifier_parameters[0]["sigmas_seed"][:,0], 1024.93939e-9, .0, 3.23076923e-9)
        self.J_pulse_out_carbide, self.E_pulse_energy_carbide, self.p_inv_out_seed_carbide, self.p_inv_out_pump_carbide, number_passes, saturation_condition = RA.run_simulation(self.amplifier_parameters[1], self.amplifier_parameters[0], self.sigmas_pump, self.sigmas_seed,notch)
        #convert to freq domain
        self.carbide_freq_intensity = self.J_pulse_out_carbide[:,number_passes-1]*(np.pi*self.amplifier_parameters[1]["radius_laser_and_pump_mode"]**2)*self.sigmas_seed[:,0]**2/(2*np.pi*c)
        temp_freq_vec = c/self.sigmas_seed[:,0]
        intensity_freq_direct_alt = np.concatenate((np.array([self.carbide_freq_intensity[0]/10]),self.carbide_freq_intensity,np.array([self.carbide_freq_intensity[-1]/10])))
        temp_freq_vec_alt = np.concatenate((np.array([np.min(self.dazzler_freq_vector)]), temp_freq_vec,np.array([np.max(self.dazzler_freq_vector)])))
        freq_inten_new=self.resample_method1(temp_freq_vec_alt, self.dazzler_freq_vector,intensity_freq_direct_alt)
        #assuming no negative values from output of carbide
        filt = np.zeros(freq_inten_new.shape[0])
        filt[1500:1700] = np.ones(200)
        #freq_inten_new_shifted = freq_inten_new-np.min(freq_inten_new)
        freq_inten_new_shifted = freq_inten_new*filt
        freq_inten_new_shifted_normalized = freq_inten_new_shifted/np.max(freq_inten_new_shifted)
        self.carbide_Efield_freq_domain_output = freq_inten_new_shifted_normalized*np.exp(-1j*np.arctan2(np.imag(self.dazzler_E_field_output_ft),np.real(self.dazzler_E_field_output_ft)))
        self.carbide_Efield_td_output = np.fft.ifft(self.carbide_Efield_freq_domain_output)
        self.carbide_Efield_td_output = self.carbide_Efield_td_output*np.exp(-1j*(carbide_phase[0]*self.dazzler_time_vector+carbide_phase[1]*self.dazzler_time_vector**2+carbide_phase[2]*self.dazzler_time_vector**3+carbide_phase[3]*self.dazzler_time_vector**4))
        plt.plot(self.dazzler_time_vector,np.abs(self.carbide_Efield_td_output)**2)
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        #probably shouldn't be here but for now
        #fix hard coding
        self.sfg_input = {"E_field":self.carbide_Efield_td_output/np.max(self.carbide_Efield_td_output),"time_vector":self.dazzler_time_vector, "frequency_vector":self.dazzler_freq_vector,'central_frequency': 299792458/(1024e-9)}

        


    #need to generalize this for crystal type eventually
    #need to allow for manual set
    def initialize_sfg(self, specphase_2nd=None, specphase_3rd=None):
        self.u = pyssnl.UNITS()
        '''
        ....
        
        '''
        time_vector = np.linspace(-8.191e-11,8.192e-11,num=16384)
        freq_vector = np.fft.fftfreq(n=time_vector.shape[0], d = (time_vector[1]-time_vector[0]))

        input_eField_ = np.sqrt(4.7*10**15)*np.exp(-1.386*(time_vector/(246*10**(-15)))**2)*np.exp(-1j*299792458/(1024e-9)*2*np.pi*time_vector)
        #adjustment = np.unwrap(np.arctan2(np.imag(input_eField_),np.real(input_eField_)))[input_eField_.shape[0]//2]
        #input_eField_ = np.sqrt(np.abs(input_eField_)**2)*(np.exp(-1j*((np.arctan2(np.imag(input_eField_),np.real(input_eField_))))-2*adjustment))
        #frankenstein_field = np.sqrt(np.abs(self.sfg_input["E_field"])**2)*np.exp(-1j*299792458/(1024e-9)*2*np.pi*self.sfg_input["time_vector"])
        frankenstein_field = np.sqrt(np.abs(self.sfg_input["E_field"])**2)*np.exp(1j*np.arctan2(np.imag(input_eField_),np.real(input_eField_))[1000:15000])

        
        #input_eField = {'time_vector':time_vector, 'E_field': input_eField_, 'frequency_vector':freq_vector, 'central_frequency': 299792458/(1024e-9) }#299792458/(input_eField2['central_wavelength'])} #frequency_vector']}
        #input_eField = {"E_field":frankenstein_field,"time_vector":self.dazzler_time_vector, "frequency_vector":self.dazzler_freq_vector,'central_frequency': 299792458/(1024e-9)}
        #print("adjust:" + str(adjustment))
        plt.plot(time_vector,np.abs(input_eField_)**2)
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        plt.plot(time_vector,np.unwrap(np.arctan2(np.imag(input_eField_),np.real(input_eField_))))
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        print("HEEYYY")
        plt.plot(self.sfg_input["time_vector"],np.abs(self.sfg_input["E_field"])**2)
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        plt.plot(self.sfg_input["time_vector"],np.unwrap(np.arctan2(np.imag(self.sfg_input["E_field"]),np.real(self.sfg_input["E_field"]))))
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        
        plt.plot(self.sfg_input["time_vector"],np.abs(frankenstein_field)**2)
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        plt.plot(self.sfg_input["time_vector"],np.unwrap(np.arctan2(np.imag(frankenstein_field),np.real(frankenstein_field))))
        plt.xlim(-.5e-12,.5e-12)
        plt.show()
        print(self.sfg_input["E_field"].shape)
        self.sfg_input = input_eField
        self.ssnl_Obj1 = pyssnl.SSNL(self.sfg_input)

        #self.ssnl_Obj1 = pyssnl.SSNL(self.sfg_input)
        #self.ssnl_Obj1.set_default(specphase_2nd_order = 2.561, specphase_3rd_order=.4)
        self.ssnl_Obj1.set_default()
        self.ssnl_Obj1.genEqns()
        self.ssnl_Obj1.genGrids()
        self.ssnl_Obj1.genFields(threshold=1e-5)
        #plt.figure()
        #plt.plot(input_eField_)
        #plt.show()

    def run_sfg(self, file_name):
        self.ssnl_Obj1.propagate()
        self.ssnl_Obj1.saveFile(file_name)

        #TIME DOMAIN
        t_long_dazzler = np.array(self.sfg_input['time_vector'])/self.u.ps
        self.t_short_dazzler = self.ssnl_Obj1.lists['t']/self.u.ps

        #FREQUENCY DOMAIN
        f_long_dazzler = np.array(self.sfg_input['frequency_vector'])
        f_short_dazzler = self.ssnl_Obj1.lists['dOmega']/(2*np.pi)

        self.dazzler_out_td = self.sfg_input['E_field']
        self.dazzler_out_fd =  np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.sqrt(np.abs(np.array(self.sfg_input['E_field']))**2))))

        self.sfg_out_time = self.ssnl_Obj1.eField['time'][3]













    


    # def initialize_sfg(self,)
    #     self.usingSFG = assertTrue
    #     u = pyssnl.UNITS()


def convert_wavelength_to_time(wavelength_vec, spectrum):
    freq_vector_temp = c/wavelength_vec
    intensity_freq_direct = spectrum*wavelength_vec**2/(2*np.pi*c)
    #intensity_freq_direct_alt = np.concatenate((np.array([intensity_freq_direct[0]/10]),intensity_freq_direct,np.array([intensity_freq_direct[-1]/10])))
    #temp_freq_vec_alt = np.concatenate((np.array([np.min(freq_vector_temp)]/10), freq_vector_temp,np.array([np.max(freq_vector_temp)])*10))
    #freq_inten_new = self.resample_method1(temp_freq_vec_alt, intensity_freq_direct_alt,)
    td_output = np.fft.ifft(intensity_freq_direct)
    return td_output

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

def construct_timeDomain_Efield(time_vec, intensity, phase):
    print("Need to implement")
    return 0
def read_in_amplifier_parameter(file_path):
    print("Need to implement")

def read_in_dazzler_parameters(file_path):
    print("Need to implement")

