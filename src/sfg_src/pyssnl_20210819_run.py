## import necessary functions
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pickle as pickle 
import pyssnl_20210819 as pyssnl


## Calling modified dazzler code

#unpickling file input from Rose:  OutputforAmy_SamplingRate240_NoShapeGaussian_7Keys --input_closeTo0AtEdges.txt OutputforAmy_SamplingRate240_NoShapeGaussian_7Keys

#files 
# LowerLambdaAmpShaping_minHoleDepth (1)
# LowerLambdaAmpShaping_medHoleDepth (1)
# LowerLambdaAmpShaping_maxHoleDepth (1)
# CentralAmpShaping_minHoleDepth (1)
# CentralAmpShaping_medHoleDepth (1)
# CentralAmpShaping_maxHoleDepth (1)
# HigherLambdaAmpShaping_minHoleDepth (1)
# HigherLambdaAmpShaping_medHoleDepth (1)
# HigherLambdaAmpShaping_maxHoleDepth (1)
# 4thPhaseShaping_maxConstant
with open('OutputforAmy_SamplingRate240_NoShapeGaussian_7Keys.txt', 'rb') as handle:
    eField= handle.read()
input_eField2 = pickle.loads(eField)

#input_eField = {'time_vector':input_eField2['time_vector'], 'E_field': input_eField2['E_field']/abs(max(input_eField2['E_field'])), 'frequency_vector':input_eField2['ang_freq_vector']/(2*np.pi), 'central_frequency': 299792458/(1030e-9)} #frequency_vector']}
input_eField = {'time_vector':input_eField2['time_vector'], 'E_field': input_eField2['E_field']/abs(max(input_eField2['E_field'])), 'frequency_vector':input_eField2['frequency_vector'], 'central_frequency': 299792458/(1030e-9) }#299792458/(input_eField2['central_wavelength'])} #frequency_vector']}



#call modified version of SFG code 
u = pyssnl.UNITS()
ssnl_Obj = pyssnl.SSNL(input_eField) #ssnl_amy.SSNL(input_pk, spec_phase_1, spec_phase_2) ,1.05*-0.1,-0.5*2.2 
ssnl_Obj.set_default() #the last two inputs are for the 2nd and 3rd order spectral phases 
ssnl_Obj.genEqns()
ssnl_Obj.genGrids() 
ssnl_Obj.genFields()
ssnl_Obj.propagate()
ssnl_Obj.saveFile('desired_file_name_for_saved peak')

#make plotting function #######################################################################################
E_field_dazzler_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(input_eField['E_field'])))
E_field_dazzler_fd_2 = np.fft.fftshift(np.fft.fft(np.fft.fftshift(input_eField2['E_field'])))

#xvals 
    #TIME DOMAIN 
t_long_dazzler = np.array(input_eField['time_vector'])/u.ps
t_short_dazzler = ssnl_Obj.lists['t']/u.ps

    #FREQUENCY DOMAIN 
f_long_dazzler = np.array(input_eField['frequency_vector'])
f_short_dazzler = ssnl_Obj.lists['dOmega']/(2*np.pi)

#comparison_plot(E_field_original_fd,f_short_original, E_field_dazzler_fd ,np.fft.fftshift( f_long_dazzler), 'Frequency', 'Comparing Frequency Space SFG and Dazzler peaks')

plt.figure()
plt.plot(input_eField['time_vector'],input_eField['E_field'], label = 'Dazzler peak input')
plt.xlabel('Time (s)')
plt.ylabel('Intensity Envelope')
plt.legend()

plt.figure()
#plt.plot(f_short_original, E_field_original_fd, label = 'SFG Self-Constructed Peak (E = 17 uJ)')
plt.plot(np.fft.fftshift(f_long_dazzler),E_field_dazzler_fd, label = 'Dazzler peak input (E = 17 uJ)')
plt.xlabel('Frequency (1/s)')
plt.ylabel(' Envelope')
plt.legend()


plt.figure()
#plt.plot(f_short_original, E_field_original_fd/max(E_field_original_fd), label = 'SFG Self-Constructed Peak (E = 17 uJ)')
plt.plot(np.fft.fftshift(f_long_dazzler),E_field_dazzler_fd/max(E_field_dazzler_fd), label = 'Dazzler peak input (E = 17 uJ)')
plt.xlabel('Frequency (1/s)')
plt.ylabel(' Envelope')
plt.title('Pre-downsampling')
plt.legend()

#shaping 

##different efields ######################################################

    #INPUT

#input_original_td = a.before['time_input']
input_dazzler_td = input_eField['E_field'] 

    #FOURIER TRANSFORM TO FREQUENCY SPACE

#input_original_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(a.before['time_input'])))
input_dazzler_fd = np.fft.fftshift(np.fft.fft(np.fft.fftshift(np.sqrt(np.abs(np.array(input_eField['E_field']))**2))))

efield_td = np.sqrt(np.abs(np.array(input_eField['E_field']))**2)

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


plt.figure()
plt.plot(t_short_dazzler,abs(ssnl_Obj.eField['time'][1][0,:])**2, color = 'b')
plt.plot(t_short_dazzler,abs(ssnl_Obj.eField['time'][2][0,:])**2, color = 'r')
plt.title('Dazzler:  Intensity of Chirped Inputs E field')
plt.xlabel('Time (ps)')
plt.ylabel('Intensity')

normalizing_const = max(max(ssnl_Obj.eField['time'][2][0,:]**2), max(ssnl_Obj.eField['time'][1][0,:])**2)
normal_d_chirp_1 = abs(ssnl_Obj.eField['time'][1][0,:])**2/normalizing_const


normal_d_chirp_2 = abs(ssnl_Obj.eField['time'][2][0,:])**2/normalizing_const
normalizing_const = max(max(ssnl_Obj.eField['time'][2][0,:]**2), max(ssnl_Obj.eField['time'][1][0,:])**2)

plt.figure()
plt.plot(t_short_dazzler, normal_d_chirp_1 , color = 'b')
plt.plot(t_short_dazzler, normal_d_chirp_2 , color = 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Intensity')


plt.figure()
plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][1][0,:]/ max(max(ssnl_Obj.eField['time'][1][0,:]),max(ssnl_Obj.eField['time'][2][0,:] )) , color = 'b')
plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][2][0,:]/ max(max(ssnl_Obj.eField['time'][2][0,:]),max(ssnl_Obj.eField['time'][1][0,:])) , color = 'r')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Intensity')


plt.figure()
#plt.plot(t_short_dazzler, ssnl_Obj.eField['time'][3][-1]/abs(max(abs(ssnl_Obj.eField['time'][3][-1]))), label = 'Normalized Real Amplitude', color = 'k')
plt.plot(t_short_dazzler, abs(ssnl_Obj.eField['time'][3][-1])**2/max(abs(ssnl_Obj.eField['time'][3][-1])**2), label = 'SFG Output', color = 'k')
plt.xlabel('Time (ps)')
plt.ylabel('Normalized Intensity')
plt.legend()
plt.ylim([-2,2])


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

wavelength_vector, I_wavelength, Phase_wavelength = convert_to_wavelength(abs(ssnl_Obj.eField['freq'][3][-1])**2, np.angle(ssnl_Obj.eField['freq'][3][-1]),ssnl_Obj.lists['omega'][2]/(2*np.pi), ssnl_Obj.lists['lambda'][2])

plt.figure()
plt.plot(wavelength_vector, I_wavelength, label = 'Intensity', color = 'b')
plt.xlabel('Wavelength (m)')
plt.ylabel('Intensity')
plt.legend()



c = 299792458
lambda_list = c/(input_eField['frequency_vector']+0.00001)
wavelength_vector, I_wavelength, Phase_wavelength = convert_to_wavelength(abs(np.fft.fftshift(np.fft.fft(np.fft.fftshift(input_eField['E_field']))))**2, np.angle(np.fft.fftshift(np.fft.fft(np.fft.fftshift(input_eField['E_field'])))),np.fft.fftshift(input_eField['frequency_vector']))

plt.figure()
plt.plot(wavelength_vector, I_wavelength/max(I_wavelength), label = 'Intensity', color = 'b')
plt.xlabel('Wavelength (m)')
plt.ylabel('Intensity')
plt.legend()