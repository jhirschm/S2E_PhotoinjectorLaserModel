#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np
import sys
import os
import re
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import pylab
import h5py
from scipy.interpolate import interp1d
import csv
import pandas as pd
import pickle
import matplotlib.colors as colors
import pypret
#from FinalCompleteMeshDataObject import complete_object

class Dazzler_Pulse_Shaper():  
    def __init__(self, position, pulsewidth, hole_position, hole_width, hole_depth, delay, sec_order, third_order, fourth_order, c = 299792458.0):
        '''
        Initializes all dazzler parameters.
        These include central wavelength and width of input field; hole position, width, and depth; and 
        dispersion factors (first, second, third, and fourth order)
        '''
        self.position = position #this is central wavelength in m
        self.pulsewidth = pulsewidth #this is in time (seconds)
        
        self.hole_position = hole_position
        self.hole_depth = hole_depth #0 to 1
        self.hole_width = hole_width
        
        #bounds listed in terms of minimum (effects not visible below) order of magnitude 
        self.delay = delay #0 to 90e-13
        self.sec_order = sec_order #-60e-26 to 60e-26
        self.third_order = third_order #80e-39 to 80e-39
        self.fourth_order = fourth_order #-220e-52 to 220e-52
        
        self.c = c
        
        #dazzler manual - multiply width by 2.5
        #"width-adjusted"
        self.width = 2.18*self.position**2/(self.c*self.pulsewidth)
        self.omega0 = 2*np.pi*self.c/self.position
        self.chi0 = self.width/(2*self.position)
        self.del_omega0 = self.omega0*(self.chi0-self.chi0**3)
        self.omega1 = 2*np.pi*self.c/self.hole_position
        self.chi1 = self.hole_width/(2*self.hole_position)
        self.del_omega1 = self.omega1*(self.chi1-self.chi1**3)/2    
    
    
    def make_gaussian_pulse(self, amplitude = 1,sampling_rate=None, time_vector=np.array([False])):
        #use position and width to make gaussian input pulse
        
        if sampling_rate==None:
            sampling_rate=240/self.pulsewidth
        
        #default time vector allows maximum delay of 9 picoseconds
        if time_vector.any()==False:
            time_vector=1/sampling_rate*np.arange(-7000,7000)
        
        EofT=amplitude*np.exp(-1.386*(time_vector/self.pulsewidth)**2)*np.exp(1j*self.omega0*time_vector)

        return time_vector, EofT
        
    def calculate_amplitude_transfer_function(self, ang_freq_vector, components_return = False):
        '''
        This function takes an angular frequency vector and uses the class parameters to 
        calculate the amplitude transfer function. This ends up being a product of 
        e^-([(w-w0)/delw0]^6) * (1-k*e^-([(w-w0)/delw0]^2))
        '''
        #calculate f and g
        f =(np.exp(-((ang_freq_vector-self.omega0)/self.del_omega0)**6))
        g = 1 - self.hole_depth*(np.exp(-((ang_freq_vector-self.omega1)/self.del_omega1)**2))
        
        # return either f,g, and A or just A, where A = f*g
        if (components_return):
            return f, g, f*g
        return f*g
    
    def calculate_phase_transfer_function(self, ang_freq_vector):
        '''
        This function calculates the phase transfer function. 
        '''
        #calculate freq difference
        omega_dif = ang_freq_vector-self.omega0
        #return h as shown in equations in text
        return -(self.delay*omega_dif + self.sec_order/2 * omega_dif**2 + self.third_order/6 * omega_dif**3 + 
                 self.fourth_order/24 * omega_dif**4)
    
    def set_saved_waveform(self, position, width, hole_position, hole_width, hole_depth,
                 delay, sec_order, third_order, fourth_order):
        
        self.savedPulse=Dazzler_Pulse_Shaper(position, width, hole_position, hole_width, hole_depth,
                 delay, sec_order, third_order, fourth_order)
        
        #to use this, a_saved in the following function must not be 0
        #see Dazzler manual or SULI Report for usage

    def calculate_full_transfer_function(self,ang_freq_vector, a_saved=0, phi_saved=0, components_return=False):
        '''
        This function calculates the full transfer function which is the product of the 
        phase and amplitude transfer functons. 

        Parameters
        ----------
        ang_freq_vector : TYPE
            DESCRIPTION.
        S_saved : TYPE, optional
            DESCRIPTION. The default is 0.
        a_saved : TYPE, optional
            DESCRIPTION. The default is 0.
        components_return : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        f,g,A_dial = None,None, None
        
        
        #calculate A dial
        if (components_return):
            f,g,A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=True)
        else:
            A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=False)
            
        #calculate phi dial
        phi_dial = self.calculate_phase_transfer_function(ang_freq_vector)
        
        if a_saved != 0:
            S_saved = self.savedPulse.calculate_full_transfer_function(ang_freq_vector)
        else:
            S_saved = 0
            
        #calculate S
        S_full = A_dial*np.exp(1j*phi_dial) + a_saved*np.exp(1j*phi_saved)*S_saved
        
        
        if (components_return):
            return f,g,A_dial,phi_dial, S_full
        
        return S_full
 
    def shape_input_pulse_low_sampling(self, E_field_input, time_vector, phi_saved=0, a_saved=0):
        
        sampling_rate=1/(time_vector[1]-time_vector[0])
        spacing=sampling_rate/len(time_vector)
        maxval=spacing*len(time_vector)/2 #max value in fftfreq-generated frequency vector
        displacement=np.abs(2*(self.omega0-maxval))
        
        freq_vector=displacement+np.fft.fftfreq(n=(time_vector.size), d=1/sampling_rate) #displace to include peak
        
        E_field_input_freq = 0.600625*self.pulsewidth*np.exp(-0.180375*(self.pulsewidth**2)*(self.omega0-(2*np.pi*freq_vector))**2)
        S_full=self.calculate_full_transfer_function(2*np.pi*freq_vector, phi_saved, a_saved)
        E_field_output_freq=S_full*E_field_input_freq
        E_field_output=np.fft.ifftshift(np.fft.ifft(E_field_output_freq))
        
        return E_field_input, E_field_input_freq, E_field_output, E_field_output_freq, time_vector, freq_vector, None
        
        #this is still very limited - only gets you down to 190/pulsewidth with accuracy. not sure why. expanding time vector doesn't help
        #with more time, I would make the "displacement" line more intelligent
        
    def shape_input_pulse(self, E_field_input, time_vector, wavelength_vec_limits=None,
                          phi_saved=0, a_saved=0, components_return=False):
        '''
        This function takes the input electric field, time vector, and sampling rate and calls
        on other functions in the class in order to calculate output electric field
        '''
        sampling_rate=1/(time_vector[1]-time_vector[0])
        if wavelength_vec_limits==None:
            wavelength_vec_limits=[self.position-200e-9, self.position+200e-9]
            
        #take FT of input electric field
        E_field_input_ft = np.fft.fft(E_field_input)
        
        #get frequency vector and angular frequency vector based on sampling rate
        freq_vector = np.fft.fftfreq(n=(E_field_input_ft.size), d=1/sampling_rate)
        ang_freq_vector = 2*np.pi*freq_vector

        #calculate transfer function
        f,g,A_dial,phi_dial, S_full = None, None, None, None, None
        if (components_return):
            f,g,A_dial,phi_dial, S_full = self.calculate_full_transfer_function(ang_freq_vector, phi_saved, a_saved, components_return)
            components_dict = {"f":f,"g":g,"A_dial":A_dial,"phi_dial":phi_dial,"S_full":S_full}

        else :
            S_full = self.calculate_full_transfer_function(ang_freq_vector, phi_saved, a_saved, components_return)
            components_dict = {"S_full":S_full}
            
        if (wavelength_vec_limits[0]<self.c/max(np.abs(freq_vector))) or (wavelength_vec_limits[1]>self.c/freq_vector[1]):
            print("Your sampling rate is not sufficient for accurate information in the frequency domain. We will recreate the input field in the frequency domain to avoid the first Fourier transform. The frequency vector will not take the typical form; some fftshifts may no longer be needed.")
            self.shape_input_pulse_low_sampling(E_field_input, time_vector, phi_saved, a_saved)
        
        #calculate output frequency domain electric field
        E_field_output_ft = E_field_input_ft*S_full
                
        #calculate IFT of output field
        E_field_output = np.fft.ifft(E_field_output_ft)
        
        
        return E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict
    
    
    

    def calculate_spectrum_phase(self, field):
        '''
        THis function separates spectrum and phase, unwrapping phase
        '''
        phase = -1*np.arctan2(np.imag(field),np.real(field))
        phase = np.unwrap(phase)
        spectrum = (np.abs(field))**2
        return spectrum, phase
    
    def convert_to_wavelength(self,I_freq, phase_freq, freq_vec, wavelength_vec_limits=None):
        '''
        This function converts the frequency domain signal into a wavelength domain. 
        Defaults to a wavelength vector that goes from 200nm below to 200nm above the central 
        wavelength. This should be adjusted if width of distribution is very large. 

        Parameters
        ----------
        I_freq : TYPE
            DESCRIPTION.
        phase_freq : TYPE
            DESCRIPTION.
        freq_vec : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.
        wavelength_vec_range : TYPE, optional
            DESCRIPTION. The default is 200e-9.

        Returns
        -------
        None.

        '''
        if wavelength_vec_limits==None:
            wavelength_vec_limits=[self.position-200e-9, self.position+200e-9]
        
        wavelength_vector = np.linspace(wavelength_vec_limits[0],wavelength_vec_limits[1],num=len(freq_vec))
        I_freq_interp = interp1d(2*np.pi*freq_vec, I_freq)
        I_wavelength = (2*np.pi*self.c/(wavelength_vector**2))*I_freq_interp(2*np.pi*self.c/wavelength_vector)
        phase_freq_interp = interp1d(2*np.pi*freq_vec, phase_freq)
        phase_wavelength = phase_freq_interp(2*np.pi*self.c/wavelength_vector)

        return wavelength_vector, I_wavelength, phase_wavelength    
        
        
    def calculate_total_spectrum_phase(self,E_field_input, E_field_input_ft, E_field_output, E_field_output_ft, S_full,
                             time_vector, freq_vector, wavelength_vector_limits=None):
        
        
        # Input 
        intensity_input_td, phase_input_td = self.calculate_spectrum_phase(E_field_input) # Time domain (td)
        spectrum_input_fd, phase_input_fd = self.calculate_spectrum_phase(E_field_input_ft) # Frequency domain (fd)
        wavelength_vector, spectrum_input_wd, phase_input_wd = self.convert_to_wavelength(spectrum_input_fd,
                                                                                     phase_input_fd,freq_vector,
                                                                                    wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd)
        
        input_functions = {"time_vector": time_vector,
                           "freq_vector":freq_vector,
                            "wavelength_vector":wavelength_vector,
                            "intensity_input_td":intensity_input_td,
                            "phase_input_td":phase_input_td,
                            "spectrum_input_fd":spectrum_input_fd,
                            "phase_input_fd":phase_input_fd,
                            "spectrum_input_wd":spectrum_input_wd,
                            "phase_input_wd":phase_input_wd}
                
        
        # Transfer function 
        S_full_td = np.fft.ifft(S_full)
        intensity_transfer_td, phase_transfer_td = self.calculate_spectrum_phase(S_full_td) # Time domain (td)
        spectrum_transfer_fd, phase_transfer_fd = self.calculate_spectrum_phase(S_full) # Frequency domain (fd)
        wavelength_vector,spectrum_transfer_wd, phase_transfer_wd = self.convert_to_wavelength(spectrum_transfer_fd,phase_transfer_fd,
                                                                        freq_vector,
                                                                        wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd) 
        
        transfer_functions = {"time_vector": time_vector,
                              "freq_vector": freq_vector,
                              "wavelength_vector": wavelength_vector, 
                              "intensity_transfer_td": intensity_transfer_td,
                              "phase_transfer_td": phase_transfer_td,
                              "spectrum_transfer_fd":spectrum_transfer_fd,
                              "phase_transfer_fd":phase_transfer_fd,
                              "spectrum_transfer_wd":spectrum_transfer_wd,
                              "phase_transfer_wd":phase_transfer_wd}
              
        # Output  
        intensity_output_td, phase_output_td = self.calculate_spectrum_phase(E_field_output) # Time domain (td)
        spectrum_output_fd, phase_output_fd = self.calculate_spectrum_phase(E_field_output_ft) # Frequency domain (fd)
        wavelength_vector, spectrum_output_wd, phase_output_wd = self.convert_to_wavelength(spectrum_output_fd,phase_output_fd,
                                                                        freq_vector,
                                                                        wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd)
        output_functions = {"time_vector": time_vector,
                            "freq_vector":freq_vector,
                            "wavelength_vector":wavelength_vector,
                            "intensity_output_td":intensity_output_td,
                            "phase_output_td":phase_output_td,
                            "spectrum_output_fd":spectrum_output_fd,
                            "phase_output_fd":phase_output_fd,
                            "spectrum_output_wd":spectrum_output_wd,
                            "phase_output_wd":phase_output_wd}
        
        return input_functions, transfer_functions, output_functions
    
    def make_plots(self, functions, domains, E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,
                             time_vector, freq_vector, wavelength_vector_limits=None):
        #put domain and type in list of string form: ["input", "transfer"]
        S_full=self.calculate_full_transfer_function(2*np.pi*freq_vector)

        input_func_fd = E_field_input_ft
        transfer_func_fd = S_full
        output_func_fd = E_field_output_ft

        input_func_td = E_field_input
        transfer_func_td = np.fft.ifft(S_full)
        output_func_td = E_field_output

        if wavelength_vector_limits==None:
                wavelength_vec_limits=[self.position-200e-9, self.position+200e-9]
                wavelength_vector = np.linspace(wavelength_vec_limits[0],wavelength_vec_limits[1],num=len(freq_vector))

        #these may be missing factors to get the intensity scaling right
        input_field_interp = interp1d(2*np.pi*freq_vector, E_field_input_ft)
        input_func_wd = input_field_interp(2*np.pi*self.c/wavelength_vector) #*...
        transfer_field_interp = interp1d(2*np.pi*freq_vector, S_full)
        transfer_func_wd = transfer_field_interp(2*np.pi*self.c/wavelength_vector)
        output_field_interp = interp1d(2*np.pi*self.c/wavelength_vector, E_field_output_ft)
        output_func_wd = output_field_interp(2*np.pi*self.c/wavelength_vector)

        input_funcs = [input_func_fd,input_func_td,input_func_wd]
        transfer_funcs = [transfer_func_fd, transfer_func_td, transfer_func_wd]
        output_funcs = [output_func_fd, output_func_td, output_func_wd]

        def plot_spectrum_phase(efield,x_vals,x_label,x_lims=None):
            intensity = abs(efield)**2
            #phase = np.unwrap(np.angle(efield))
            phase = np.unwrap(-1*np.arctan2(np.imag(efield), np.real(efield)))
            xaxis = x_vals
            fig,ax1 = plt.subplots()
            color1 = 'tab:red'
            ax1.set_xlabel(x_label)
            ax1.set_ylabel('Phase', color = color1)
            ax1.plot(xaxis, phase, color = color1)
            ax1.tick_params(axis = 'y',labelcolor = color1)
            ax2 = ax1.twinx()
            color2 = 'tab:blue'
            if x_lims!=None:
                ax2.set_xlim(x_lims[0],x_lims[1])
            ax2.set_ylabel('Intensity', color = color2)
            ax2.plot(xaxis,intensity,color = color2)
            ax2.tick_params(axis = 'y', labelcolor = color2)
            fig.tight_layout()
            plt.show()

        domain_to_xval={"time": time_vector,
                       "frequency": freq_vector,
                       "wavelength": wavelength_vector}
        function_to_fieldset={"input": input_funcs,
                          "transfer": transfer_funcs,
                          "output": output_funcs}
        domain_in_fieldset={"time": 1,
                           "frequency": 0,
                           "wavelength": 2}
        for word1 in functions:
            for word2 in domains:
                plot_spectrum_phase(function_to_fieldset[word1][domain_in_fieldset[word2]],domain_to_xval[word2],word2, x_lims=None)


    def output_SFGcode(self, time_vector, E_field_output, freq_vector, filename): 
        
        parameters={"position": self.position,
               "pulsewidth": self.pulsewidth,
               "hole_position": self.hole_position,
               "hole_width": self.hole_width,
               "hole_depth": self.hole_depth,
               "delay": self.delay,
               "sec_order": self.sec_order,
               "third_order": self.third_order,
               "fourth_order": self.fourth_order}
        characterization={"Parameters": parameters,
                "E_field": E_field_output,
                "time_vector": time_vector,
                 "freq_vector": freq_vector,
                "central_wavelength": self.position}

        file1 = open(filename,"wb")
        pickle.dump(characterization, file1)
        file1.close()
        
        
    def make_FROG_trace(self, eField, time_vector, x_lims=None, y_lims=None, spectrogram=True):
        time_length_expanded = 3*len(time_vector)
        time_spacing = time_vector[1]-time_vector[0]
        sampling_rate=1/time_spacing
        time_vector_expanded = time_spacing*np.arange(-time_length_expanded/2, time_length_expanded/2)
        time_vector_expanded, eField_expanded = self.make_gaussian_pulse(self.pulsewidth, time_vector=time_vector_expanded)
        #x_lims, y_lims = None, None

        def EofTGate(field, time_vec, tau):
            returnval=[]
            i1=tau*sampling_rate #make integer #this is the amount of indices that our tau displaces the time vector
            num=len(time_vec)
            for i2 in range(num):
                index=int(num+i2-i1)
                out=np.abs((field)[index])**2
                returnval.append(out)
            return np.array(returnval)

        taulist=time_vector
        intensityarray=np.zeros((len(time_vector),len(time_vector)))
        i=0
        for tau in taulist:
            intensity_FROG=(np.fft.fft(eField*EofTGate(eField_expanded, time_vector, tau)))**2 #this gives 1D array of all frequency for given tau
            intensityarray[:,i]=np.fft.fftshift(np.abs(intensity_FROG))
            i+=1
        freq_vector = np.fft.fftshift(np.fft.fftfreq(n=intensity_FROG.size, d=1/sampling_rate))
        
        if spectrogram == True:
            ourspec=pypret.MeshData(np.transpose(intensityarray), time_vector, freq_vector)
            pypret.MeshData.axes=time_vector, freq_vector
            pypret.MeshData.labels=["time", "frequency"]
            pypret.MeshData.units=["s", "Hz"]
            if y_lims!=None:
                if x_lims!=None:
                    ourspec.limit(x_lims, y_lims)
                else:
                    ourspec.limit([-2e-12,2e-12], y_lims)
            pypret.MeshDataPlot(ourspec)
            
            '''
            my (much slower) code to generate spectrograms:
            
            if x_lims!=None:
                plt.xlim(x_lims[0],x_lims[1])
            if y_lims!=None:
                plt.ylim(y_lims[0],y_lims[1])
            plt.pcolormesh(taulist,freq_vector,intensityarray, norm=colors.SymLogNorm(linthresh=20, linscale=1, vmin=np.amin(intensityarray), vmax=np.amax(intensityarray), base=np.e), shading='gouraud')
            plt.colorbar()
            plt.show()
            '''
        
        return intensityarray
    

'''
    def FROG_retrieval(intensityarray, time_vector, freq_vector, E_field_input_guess, E_field_output_ft=None):
        #feed intensityarray into pypret retrieval function
        #if efield!=None, compare it with retrieved efield by plotting on one graph
                
        complete_object(np.transpose(intensityarray), time_vector, freq_vector)
        
        ret = pypret.Retriever(ourspec, "copra", verbose=True, maxiter=30)
        ret.retrieve(ourspec, E_field_input) #E_field_input is the initial "guess"
        data=ret.result()
        E_field_retrieved_freq = ret._result.spectrum #gives frequency spectrum
        
        if E_field_output!=None:
            plt.plot(time_vector, np.abs(E_field_retrieved)**2)
            plt.plot(time_vector, np.abs(E_field_output_ft)**2)
            
        return E_field_retrieved_freq
    
        #this has worked well several times but is still persnickety after transfer from Spyder!! however, I hope it's a good starting place
        #another unanswered question is preservation of intensity magnitude 
'''
