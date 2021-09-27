#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 12:00:15 2021

@author: student
"""

import FinalDazzlerClass
import FinalCompletetheMeshDataObject

#main functionalities: make pulse, shape pulse, make FROG spectrogram
#initialize class with two parameters describing central wavelength and pulsewidth, 3 parameters describing hole, 4 parameters for phase shaping
trial = Dazzler_Pulse_Shaper(1030e-9,330e-15,1030e-9,4e-9,1,0,0e-26,0,0)
#create input pulse in time domain: optional arguments amplitude, sampling rate, and time vector
time_vector, EofT = trial.make_gaussian_pulse()
#shape pulse: only requires Dazzler class initialization and the 2 arguments that are outputs of make_gaussian_pulse
E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict = trial.shape_input_pulse(EofT, time_vector)

'''
make_plots
'''
trial.make_plots(["input", "transfer", "output"], ["time", "frequency", "wavelength"], E_field_input, E_field_input_ft, E_field_output, E_field_output_ft, time_vector, freq_vector)
#this utility is very helpful (you can see any combination of the above), but phase for frequency input and output should be fftshifted as should intensity for time transfer- no good way found to change just these 


'''
SFG_output
'''
#creates the agreed upon SFG output to the file specified 
trial.output_SFGcode(time_vector, E_field_output, freq_vector, "trialoutput.txt")


'''
set saved waveform
'''
#to set saved waveform, you must first set its parameters using "set_saved_waveform", then also weight it in the "shape_input_pulse" function.
#here we create an initial waveform with delay and hole, then a 2nd waveform with greater delay.
separate_trial=Dazzler_Pulse_Shaper(1030e-9,330e-15,1030e-9,4e-9,1,2e-12,0,0,0)
time_vector_s, EofT_s = separate_trial.make_gaussian_pulse()
separate_trial.set_saved_waveform(1030e-9,330e-15,1030e-9,4e-9,0,6e-12,0,0,0)
E_field_input_s, E_field_input_ft_s, E_field_output_s, E_field_output_ft_s,time_vector_s, freq_vector_s, components_dict_s = separate_trial.shape_input_pulse(EofT_s, time_vector_s, phi_saved=.5, a_saved=1)
#see Dazzler software interface: these a_saved and phi_saved correspond to the "amp" and "phase" controls which can be arrowed up and down
plt.grid()
plt.plot(time_vector_s, E_field_output_s)
#a_saved represents ratio of saved peak height to entered peak height (not accounting for amplitude shape)
#phi_saved obviously has no affect on real-valued intensity, but looking at complex E-field, it "stretches/compresses" tops and bottoms of each peak


'''
slightly lower sampling rate bound (shape_input_pulse_low_sampling)
'''
#we found that the Fourier Transform from time to frequency only works down to sampling_rate of 240/our_pulsewidth (7.27e14).
#here is a workaround that can go down to 195/our_pulsewidth = 5.91e14 (this cutoff is independent of pulsewidth, and not sure why it stops working there).
#the user doesn't have to do anything extra: workaround is nested in more general function and will provide warning.
sampling_trial=Dazzler_Pulse_Shaper(1030e-9,240e-15,1030e-9,5e-9,1,5e-12,0,0,0)
time_vector_st, EofT_st = sampling_trial.make_gaussian_pulse(sampling_rate=5.91e14)
E_field_input_st, E_field_input_ft_st, E_field_output_st, E_field_output_ft_st,time_vector_st, freq_vector_st, components_dict_st = sampling_trial.shape_input_pulse(EofT_st, time_vector_st)
plt.plot(time_vector_st, E_field_output_st)


'''
make spectrogram
''' 
#only requires field output in time domain, and time vector. Optional x_lims and y_lims, filename to dump 2D intensity array (since generating it can take several minutes)
#for the purpose of time efficiency I'm going to make a smaller time_vector to do this
#the spectrogram will come out pixel-y, but in less than 10 seconds: can decide this trade-off, in conjunction with needs of parameters
time_vector_cut=trial.pulsewidth/240*np.arange(-800,800)
time_vector, EofT = trial.make_gaussian_pulse(time_vector=time_vector_cut)
E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict = trial.shape_input_pulse(EofT, time_vector_cut)
intensityarray=trial.make_FROG_trace(E_field_output,time_vector, y_lims=[2.81e14,3.01e14], filename="newIA.txt")

'''check FROG retrieval
'''
#this just returns the retrieved pulse, unless you include "E_field_output_ft" in which case it also makes a graph of the two overlaid on each other in frequency domain
trial.FROG_retrieval(intensityarray, time_vector_cut, freq_vector, E_field_output_ft=E_field_output_ft, x_lim=[2.81e14,3.01e14])
