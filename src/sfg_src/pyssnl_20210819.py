from contextlib import nullcontext
import numpy as np
import sympy as sp
import scipy.interpolate
import matplotlib.pyplot as plt
from numpy.fft import fftshift, ifftshift

#Necessary functions to be used in SSNL class

def fft(field):
    '''fft with shift
    
    Shifting values so that initial time is 0.
    Then perform FFT, then shift 0 back to center.
    
    field: 1xN numpy array
    
    return a 1xN numpy array'''
    return fftshift(np.fft.fft(ifftshift(field)))

def ifft(field):
    '''ifft with shift
        
    Shifting values so that initial time is 0.
    Then perform IFFT, then shift 0 back to center.
    
    field: 1xN numpy array
        
    return a 1xN numpy array'''
    return fftshift(np.fft.ifft(ifftshift(field)))

def peak_range_finder(field,threshold=1e-15):
    '''Determines range of indices over which the field has an intensity 
    higher than a set threshold value (1x10^-15 * the maximum intensity). 
    
    field: 1xN numpy array 
    
    return a 1xN numpy array containing the indices of the thresholded intensity '''
    
    intensity  = abs(field)**2
    min_intensity = threshold*np.amax(intensity)
    intensity_range_indices = np.where(intensity >= min_intensity)

    return intensity_range_indices

def make_filter(field, intensity_range, type_of_filt = 'Hanning'):
    '''Makes filter to smooth edges of peak to zero before index rotation

    field: 1xN numpy array
    intensity_range: 1xM array - output of peak_range_finder(field)
    type_of_filt: string specifying filter type out pf {'hann', 'Tukey'}
    https://en.wikipedia.org/wiki/Window_function

    return a 1xN numpy array 
    '''
    num_pts = len(field)
    window_length = len(intensity_range[0])
    #generate filter 
    if type_of_filt == 'Hanning':        
        region_filt = 0.5 - (0.5 * np.cos(2 * np.pi / window_length * np.arange(window_length))) 
    elif type_of_filt == 'Tukey': 
        region_filt = np.zeros(window_length)
        a = 0.5
        breakpnt = int(np.floor((a*window_length)/2))
        region_1 = np.arange(0,breakpnt)
        region_filt[0:breakpnt] = 0.5*(1 - np.cos((2*np.pi*region_1)/(np.floor(a*window_length))))
        region_2 = np.zeros(int(window_length/2)-breakpnt) + 1
        region_filt[breakpnt:int(window_length/2)] = region_2
        region_3 = np.flip(region_filt[0:int(window_length/2)])
        region_filt[int(window_length/2):int(window_length/2)+len(region_filt[0:int(window_length/2)])] = region_3
        
    
    space_on_right = num_pts - intensity_range[0][-1]
    space_on_left = intensity_range[0][0] -1 

    if (space_on_left<0): #added by Jack because value was negative and caused issues with np.pad
        space_on_left=0
        
    final_filt = np.pad(region_filt, (space_on_left,space_on_right), 'constant')
    return final_filt

def rotate_peak_indices(np_array, desired_pk_indx, freq_vec, central_freq): 
    '''Rotates peak so that it can be centered at f or t = 0 following a
    fourier transformation. This is acheived when desired_pk_indx = mid_pt_indx    
    np_array: 1xN numpy array with a local maximum 
    desired_pk_indx: integer index
    
    return a 1xN numpy array rotated to place the peak at desired_pk_indx'''
    centr_pk_indx = np.argmin(abs(freq_vec - central_freq))
    new_np_array = np.zeros(len(np_array))+0j

    if centr_pk_indx - desired_pk_indx < 0: #rotate counterclockwise

        num_indx_2_rotate = desired_pk_indx- centr_pk_indx
        new_np_array[0:num_indx_2_rotate-1] = np_array[len(np_array) - num_indx_2_rotate:-1]
        new_np_array[num_indx_2_rotate:-1] = np_array[0:len(np_array) - (num_indx_2_rotate+1)]

    elif centr_pk_indx - desired_pk_indx > 0:  

        num_indx_2_rotate = centr_pk_indx - desired_pk_indx
        new_np_array[len(new_np_array)-num_indx_2_rotate: -1] = np_array[0:num_indx_2_rotate-1]
        new_np_array[0:len(new_np_array)-(num_indx_2_rotate+1)] = np_array[num_indx_2_rotate:-1]

    else: 

        new_np_array = np_array

    return new_np_array 


def interpolate_Efield(efield_fd, freq_vector_old,freq_vector_new):
    '''Interpolates Efield from the dazzler-set sampling rate and vector length defined 
    by freq_vector_old to the new sampling rate and vector length defined by freq_vector_new. 
    This interpolation is necessary to preserve phase information and frequency bandwidth.
    
    efield_fd: 1xN complex numpy array 
    freq_vector_old: 1xN numpy array that was defined by self.input_freq_vector
    freq_vector_new: 1xN numpy array that was defined in genGrids and has a set length of 2**15 and a spacing of 16.5 fs
    
    return a 1xN numpy array of the input field with the sample sampling rate and length as freq_vector_new
    
    SPECIAL NOTATION: It might seem like breaking down exp(i*phase) into it's real and imaginary components 
    and then interpolating is redundant. HOWEVER, if you just interpolate the phase and then do exp(i (interpolated phase))
     there will be odd gaussian shaped growths on the exp(i phase) graph. To check this, plot exp(i (interpolated phase)) against frequency
    
    '''
    phase_fd = np.unwrap(np.angle(efield_fd), discont = 50)
    exp_phase_fd = np.exp(1j*phase_fd)
    #interpolate amplitude
    amplitude = efield_fd/np.exp(1j*phase_fd)
    amplitude_new = scipy.interpolate.griddata(freq_vector_old,amplitude,freq_vector_new, method = 'linear')
    #interpolate real part of phase - see documentation for why this is necessary
    real_exp_phase_fd = np.real(exp_phase_fd)
    real_exp_phase_fd_new = scipy.interpolate.griddata(freq_vector_old,real_exp_phase_fd ,freq_vector_new, method = 'linear')
    #interpolate imag part of phase - see documentation for why this is necessary
    imag_exp_phase_fd = np.imag(exp_phase_fd)
    imag_exp_phase_fd_new = scipy.interpolate.griddata(freq_vector_old,imag_exp_phase_fd,freq_vector_new, method = 'linear')
    exp_phase_fd_new = real_exp_phase_fd_new  + (1j*imag_exp_phase_fd_new)
    efield_new = amplitude_new*exp_phase_fd_new
    return efield_new

class UNITS:
    
    def __init__(self,mScale=0,sScale=0):
        
        self.m = 10**mScale
        self.mm = 10**(-3*self.m)
        self.um = 10**(-6*self.m)
        self.nm = 10**(-9*self.m)
        
        self.s = 10**sScale
        self.ns = 10**(-9*self.s)
        self.ps = 10**(-12*self.s)
        self.fs = 10**(-15*self.s)
        
        self.J = (self.m**2)/(self.s**2)
        self.mJ = 10**(-3*self.J)
        self.uJ = 10**(-6*self.J)

class SSNL:
    
    def __init__(self, input_eField):

        #vector peak input
        self.input_eField = input_eField['E_field']   # in the time domain 
        self.input_time_vector = input_eField['time_vector']     
        self.input_freq_vector = input_eField['frequency_vector'] # NOT angular frequency
        self.input_central_freq = input_eField['central_frequency']

        #verify form of inputs - sometimes output from dazzler was not a np array
        if isinstance(self.input_eField, np.ndarray) == False: 
            print('E field was converted to a np.ndarray')
            self.input_eField = np.array(self.input_eField)
        if isinstance(self.input_time_vector, np.ndarray) == False: 
            print('Input time vector was converted to a np.ndarray')
            self.input_time_vector = np.array(self.input_time_vector)
        if isinstance(self.input_freq_vector, np.ndarray) == False: 
            print('Input frequency vector was converted to an np.ndarray')
            self.input_freq_vector = np.array(self.input_freq_vector)
        
        #see if frequency vector needs to be shifted - sometimes frequency output from dazzler were not in ascending order
        if min(self.input_freq_vector) != self.input_freq_vector[0]:
            self.input_freq_vector = np.fft.fftshift(self.input_freq_vector)
        
        #constants and units
        u = UNITS()
        self.c          = 299792458 * (u.m/u.s)
        self.eps0       = (8.854187817 * 10**-12) / u.m
        
        #maybe later - assign all of these using the input_pk dictionary? i
        self.lams       = None
        self.ks         = None
        self.omegas     = None
        self.crys       = None
        self.length     = None
        self.theta      = None
        self.mixType    = None
        self.taus       = None
        self.energies   = None
        self.specPhases = None

    def set_default(self,  specphase_2nd_order = -0.5*2.2, specphase_3rd_order=1.05*-0.1):
        '''Set properties to case with: 
        1030 nm = fundamental wavelength 
        515  nm = second harmonic
        with a pulsewidth of 330  fs to get a squarish pulse '''

        u     = UNITS() 

    ##Initial 2nd and 3rd order dispersion constants
               
        self.tay12 = specphase_2nd_order
        self.tay13 = specphase_3rd_order

    ##Input vairables : wavelengths, frequency, and wavevector(const) of two incoming beams + SFG beam
        laser_wavelength = (self.c/self.input_central_freq)#was 1030 nm
        self.lams     = np.array([laser_wavelength, laser_wavelength, laser_wavelength*0.5])
        self.ks       = (2*np.pi)/self.lams
        self.omegas   = self.c * self.ks  #m/s x 1/m = 1/s   
        self.specPhases = np.array([
                          [-self.tay12*u.ps**2,self.tay13*u.ps**3,0,0], 
                          [self.tay12*u.ps**2,-self.tay13*u.ps**3,0,0],
                          [0,0,0,0]
                          ]) #edit (i dont think the change i made is right) made a fix to location of zero in vector, place at beginning since we are setting SOD and TOD

    ##Crystal System variables 
        self.crys     = 'BBO'
        self.length   = .5*u.mm#2*u.mm #0.5 mm
        self.theta    = 23.29 # for BBO
        self.mixType  = 'SFG'
           
        return    


    def genEqns(self,crysName=None):
        '''Creates the anonymous functions for the index of refraction, nonlinear mixing,
        taylor expansion of phase, and the derivative of 'k' for the speed of the grids.
        
        crysName: (OPTIONAL) a string. Name of the nonlinear crystal to use.
        DOES NOT WORK RIGHT NOW AS BBO IS THE ONLY INCLUDED ONE
        
        returns nothing but sets internal attributes
        '''
        if crysName is None: # Future support for other crystals
            crysName = self.crys

        u = UNITS()
        if crysName == 'BBO': 

            const_A_o,const_B_o,const_C_o,const_D_o = 2.7359,0.01878,- 0.01822,- 0.01354
            const_A_ex,const_B_ex,const_C_ex,const_D_ex  = 2.3753,0.01224,- 0.01667,- 0.01516

            dNL = 2.01e-12
            self.theta = self.theta
            
        elif crysName == 'KDP':
            self.crys = crysName

            #https://aip.scitation.org/doi/full/10.1063/1.4832225
            const_A_o, const_B_o, const_C_o, const_D_o, const_E_o = 2.25881, 0.01041, 0.01209, 11.86370, 400
            const_A_ex, const_B_ex, const_C_ex, const_D_ex, const_E_ex = 2.13338, 0.00873, 0.01203, 2.93795, 400
            
            dNL = 0.38e-12 #https://web.stanford.edu/~rlbyer/PDF_AllPubs/1990/240.pdf
            self.theta = 41.188 #http://toolbox.lightcon.com/tools/PMangles/
            
        
        (l, theta, w, lCtr, field1, field2, field3, dOmega, kk2, kk3, kk4, kk5)\
            = sp.symbols('l theta w lCtr field1 field2 field3 dOmega kk2 kk3 kk4 kk5')
        
        if crysName == 'BBO':
            nO_SYMPY = sp.sqrt( const_A_o + const_B_o/((l/u.um)**2 + const_C_o) + const_D_o * (l/u.um)**2 )
            nO = lambda l:np.sqrt( const_A_o + const_B_o/((l/u.um)**2 + const_C_o) + const_D_o * (l/u.um)**2 )
            nE = lambda l:np.sqrt( const_A_ex + const_B_ex/((l/u.um)**2 + const_C_ex) + const_D_ex * (l/u.um)**2)
        else: #sellmeier equation
            nO_SYMPY = sp.sqrt(const_A_o + const_B_o/(((l/1e-6)**2)- const_C_o) + ((const_D_o*((l/1e-6)**2))/(((l/1e-6)**2)-const_E_o)))
            nO = lambda l: np.sqrt(const_A_o + const_B_o/(((l/1e-6)**2)- const_C_o) + ((const_D_o*((l/1e-6)**2))/(((l/1e-6)**2)-const_E_o)))
            nE = lambda l: np.sqrt(const_A_ex + const_B_ex/(((l/1e-6)**2)- const_C_ex) + ((const_D_ex*((l/1e-6)**2))/(((l/1e-6)**2)-const_E_ex)))

        nE_Theta = lambda l, theta:np.sqrt( 1 / (
            np.cos(np.deg2rad(theta))**2/nO(l)**2 +
            np.sin(np.deg2rad(theta))**2/nE(l)**2
            ))
        
        self.eqns = {'index':None, 'dk':None, 'nonLin':None, 'phase':None}
                
        self.eqns['index'] = np.array((nO,nO,nE_Theta))
        
        k1 = (w/self.c)*nO_SYMPY.subs(l,(2*np.pi*self.c)/w)
        dk1 = sp.diff(k1,w)
        self.eqns['dk'] = float(dk1.subs(w,self.omegas[0]).evalf())
        
        nonLinCoef = (((dNL * 1j) * 2 * self.ks[0])/self.eqns['index'][0](self.lams[0]),
                      ((dNL * 1j) * 2 * self.ks[1])/self.eqns['index'][1](self.lams[1]),
                      ((dNL * 1j) * 2 * self.ks[2])/self.eqns['index'][2](self.lams[2],self.theta),
                      )
        
        self.eqns['nonLin'] = np.array(((lambda field2, field3: nonLinCoef[0] * np.conj(field2) * field3),
                               (lambda field1, field3: nonLinCoef[1] * np.conj(field1) * field3),
                               (lambda field1, field2: nonLinCoef[2] * field1 * field2),
                               ))
        
        dOmega = lambda l, lCtr:(2*np.pi*self.c) * ( (1/lCtr) - (1.0/l) )
        self.eqns['phase'] = lambda kk2, kk3, kk4, kk5, l, lCtr:(
            ( (kk2/np.math.factorial(2)) * (dOmega(l,lCtr)**2) ) +
            ( (kk3/np.math.factorial(3)) * (dOmega(l,lCtr)**3) ) +
            ( (kk4/np.math.factorial(4)) * (dOmega(l,lCtr)**4) ) +
            ( (kk5/np.math.factorial(5)) * (dOmega(l,lCtr)**5) )
            )
        
        pass

                
    def genGrids(self,nPts = None, dt=None,nZ=100):
        '''Creates the .grids and .lists attributes of the object for the run.
        The .grids attribute holds the info for the discrete step spacing of
        quantities such as time and space. The .lists property holds all the
        points used in computation based on the spacing from .grids and
        values from self.properties. These values will be used during the interpolation step. 
        
        nPts: (OPTIONAL) a single integer. Number of points in lists. 
                You will regret everything if it is not apower of 2.
                DEFAULT: 2**14
        dt: (OPTIONAL) a single integer. The spacing in time but also defines
                the frequency resolution. More time, tighter resolution of nPts
                around the central frequencies 
                DEFAULT: tau[1]/10
        nZ: (OPTIONAL) a single integer. Number of steps to take through the
                simulation. Higher numbers result in more accurate simulations
                but take more time linearly
                DEFAULT: 100
                
        returns nothing but sets internal attributes
        '''
        #Check inputs
        u = UNITS()
        if dt is None:
            dt = 16.5*u.fs # dt = 1/sampling rate
        else: 
            print('Sampling rate differs from standardized values. Ensure that Parseval''s theorem is being obeyed.')
            
        if nPts is None:
            nPts = 2**15 #prevents negative values for wavelength
        else:
            print('Vector differs from standardized values. Ensure that Parseval''s theorem is being obeyed.')
        
        gridKeys = ['nPts','dt','dz','nZ','dw']
        listKeys = ['t','lambda','omega','dOmega','k']
        nFields = len(self.lams)

        self.grids = {key:None for key in gridKeys}
        self.lists = {key:None for key in listKeys}
            
        self.grids['nPts'] = nPts #number of points in time, frequency, lambda, and k vectors
        self.grids['dt'] = dt  #spacing in time domain
        self.grids['nZ'] = nZ #number of steps through the crystal
        self.grids['dz'] = self.length / (self.grids['nZ'] - 1) #spacing along the z direction
        self.grids['dw'] = (2*np.pi) / (self.grids['nPts'] * self.grids['dt']) #spacing in frequency domain
        
        self.lists['t'] = self.grids['dt'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1) #time vector which DIFFERS from the input time vector
        self.lists['dOmega'] = self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1) #angular frequency vector DIFFERS in sampling rate and length from input

        #preallocating space
        self.lists['lambda'] = np.zeros((nFields,self.grids['nPts']))
        self.lists['omega'] = np.zeros((nFields,self.grids['nPts']))
        self.lists['k'] = np.zeros((nFields,self.grids['nPts']))
        

        for ii in range(nFields):
            
            self.lists['omega'][ii,:] = self.lists['dOmega'] + self.omegas[ii] #center peak at the central frequency
            self.lists['lambda'][ii,:] = np.divide(2*np.pi*self.c,self.lists['omega'][ii,:]) #center peak at the central wavelength
            
            if ii != nFields-1:
                self.lists['k'][ii,:] = (
                    np.divide(2*np.pi,self.lists['lambda'][ii,:]) *
                     self.eqns['index'][ii](self.lists['lambda'][ii,:]) - 
                      (self.lists['dOmega']*self.eqns['dk']
                       )
                     )
            elif ii == nFields-1:
                self.lists['k'][ii,:] = (
                    np.divide(2*np.pi,self.lists['lambda'][ii,:]) *
                     self.eqns['index'][ii](self.lists['lambda'][ii,:],self.theta) - 
                      (self.lists['dOmega']*self.eqns['dk']
                       )
                     )
        
        return
    
    
    
    def genFields(self,threshold=1e-15):
        '''Creates the field variables and allocates memory. This is based on all
        the attributes input and generated before hand.
        
        returns nothing but sets internal attributes        
        '''
        
        nFields = len(self.lams)
        
        timeField  = {(ii+1):
                   np.zeros((self.grids['nZ']+1,self.grids['nPts']),dtype=complex) 
                   for ii in range(nFields)
                   }
        freqField  = {(ii+1):
                   np.zeros((self.grids['nZ']+1,self.grids['nPts']),dtype=complex) 
                   for ii in range(nFields)
                   }
            
        self.eField  = {'time':timeField, 'freq':freqField}
        
        for ii in range(nFields):
            
            if ii != nFields-1:
                
                #import peaks from dazzler 
                Input_eField_td =self.input_eField
                #plt.plot(np.abs(Input_eField_td)**2)
                #plt.show()
                Input_eField_fd= fft(Input_eField_td) #will shift to a central frequency if input is oscillatory

                #make filter: hann or rectangular or tukey
                peak_intensity_range= peak_range_finder(Input_eField_fd,threshold=threshold)
                filter = make_filter(Input_eField_fd, peak_intensity_range, 'Tukey')
                #apply filter
                Input_eField_fd *= filter

                #center peak
                center_indx = int(np.round(len(self.input_freq_vector)/2))            
                Input_eField_fd = rotate_peak_indices(Input_eField_fd, center_indx, self.input_freq_vector, self.input_central_freq)

                #interpolate - effectively downsampling in the time domain
                freq_vector_old = self.input_freq_vector #input frequency vector
                freq_vector_new = self.lists['dOmega']/(2*np.pi) #new generated frequency vector
                #DRASTIC CHANGE: 1) number of elements changed 2) sampling of signal changed 3) window size changed
                self.eField['freq'][ii+1][0,:] = interpolate_Efield(Input_eField_fd, freq_vector_old, freq_vector_new)

                #stretch or compress
                self.eField['freq'][ii+1][0,:] *=  (
                np.exp( 1j * self.eqns['phase'](self.specPhases[ii,0],
                                                self.specPhases[ii,1],
                                                self.specPhases[ii,2],
                                                self.specPhases[ii,3],
                                                self.lists['lambda'][ii,:],
                                                self.lams[ii]
                                                )
                       )
                )
                
                self.eField['time'][ii+1][0,:] = ifft(self.eField['freq'][ii+1][0,:])
                
            elif ii == nFields-1:
            
                self.eField['time'][ii+1][0,:] = np.zeros(len(self.lists['t']))                    
               
        return

    
    def RKstep(self, zStep):
        '''Custom Runga-Kutta 4 algorithm to work with class structure.
        
        zStep: a single integer. Index of which step in propagation we are on.
                Is used to index the .efield property so between 1 and .grids['nZ']
        
        returns nothing but sets internal attributes
        '''
        
        nFields = len(self.lams)
        N = self.grids['nPts']
        
        if nFields == 3:
            fieldMap = np.array([[2,3],[1,3],[1,2]])
            
        rk0 = np.zeros((nFields,N),dtype=complex)
        rk1 = np.zeros((nFields,N),dtype=complex)
        rk2 = np.zeros((nFields,N),dtype=complex)
        rk3 = np.zeros((nFields,N),dtype=complex)
        x =1
        for ii in range(nFields):
            rk0[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:],
                self.eField['time'][fieldMap[ii,1]][zStep,:]
                )
        x = 1    
        for ii in range(nFields):
            
            rk1[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk0[fieldMap[ii,0]-1,:]/2,
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk0[fieldMap[ii,1]-1,:]/2
                )
        x = 1    
        for ii in range(nFields):
            rk2[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk1[fieldMap[ii,0]-1,:]/2,
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk1[fieldMap[ii,1]-1,:]/2
                )
        x = 1   
        for ii in range(nFields):
            rk3[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk2[fieldMap[ii,0]-1,:],
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk2[fieldMap[ii,1]-1,:]
                )
        x = 1
        for ii in range(nFields):
            self.eField['time'][ii+1][zStep,:] = (
                self.eField['time'][ii+1][zStep,:] +
                rk0[ii,:]/6 + rk1[ii,:]/3 + rk2[ii,:]/3 + rk3[ii,:]/6
                )
        
        
        return

    
    def propagate(self):
        '''Propagate the field along the crystal and update the .eField property.
        Each step and field is held in memory so the zStep that we are on is the
        first index in the .eField['time'] or .eField['time'] property 
        
        
        returns nothing but sets internal attributes
        '''
        
        def dzStep(self, iz):
            if iz == 1 or iz == self.grids['nZ']:
                return self.grids['dz']/2
            else:
                return self.grids['dz']
        
        nFields = len(self.lams)
        
        for iZ in range(1,self.grids['nZ']+1):
            
            for iF in range (nFields):
                self.eField['time'][iF+1][iZ,:] = ifft( 
                    self.eField['freq'][iF+1][iZ-1,:] *
                    np.exp(1j * self.lists['k'][iF,:] * dzStep(self,iZ))
                    )
            x = 1   
            if iZ <= self.grids['nZ']-1:
                
                self.RKstep(iZ)
            x = 1  
            for iF in range(nFields):
                self.eField['freq'][iF+1][iZ,:] = fft(
                    self.eField['time'][iF+1][iZ,:]
                    )
        return
    
    
    def saveFile(self, filename):
        import pickle as pickle 
        pickle_filename = filename + '.txt'

        #dictionary containing all of the SFG parameters
        SFG_params_keys = ['2nd_Order_Dispersion','3rd_Order_Dispersion','Crystal_type','Crystal_length','Central_wavelength']
        SFG_params = {key:None for key in SFG_params_keys}
        
        SFG_params['2nd_Order_Dispersion'] =  self.tay12
        SFG_params['3rd_Order_Dispersion'] =  self.tay13
        SFG_params['Crystal_type'] =  self.crys
        SFG_params['Crystal_length'] =  self.length
        SFG_params['Central_wavelength'] = self.lams[2]

        #Final output containing peak information
        Output_keys = ['E_field', 'time_vector', 'freq_vector', 'central_wavelength', 'SFG_params']
        Output = {key:None for key in Output_keys}
        
        Output['E_field'] = self.eField['freq'][3][-1]
        Output['time_vector'] = self.lists['t']
        Output['freq_vector'] = self.lists['omega'][2]/(2*np.pi)
        Output['central_wavelength'] = self.lams[2]
        Output['SFG_params'] = SFG_params            

        with open(pickle_filename, 'wb') as handle:
            pickle.dump(Output, handle, protocol=pickle.HIGHEST_PROTOCOL)
