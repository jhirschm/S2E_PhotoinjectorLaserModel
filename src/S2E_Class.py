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

class S2E():
    def __init__(self):

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

        self.dazzlerObject = DazzlerClass.Dazzler_Pulse_Shaper(self.central_wavelength,self.width,self.hole_position,self.hole_width,self.hole_depth,self.delay,self.second_order,self.third_order,self.fourth_order)
        if(selfGenerateInput):
            self.input_time_vector, self.Efield_input = dazzlerObject.make_gaussian_pulse()
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


        self.dazzlerObject.position = position #this is central wavelength in m
        self.dazzlerObject.pulsewidth = pulsewidth #this is in time (seconds)

        self.dazzlerObject.hole_position = hole_position
        self.dazzlerObject.hole_depth = hole_depth #0 to 1
        self.dazzlerObject.hole_width = hole_width

        self.dazzlerObject.delay = delay #0 to 90e-13
        self.dazzlerObject.sec_order = sec_order #-60e-26 to 60e-26
        self.dazzlerObject.third_order = third_order #80e-39 to 80e-39
        self.dazzlerObject.fourth_order = fourth_order #-220e-52 to 220e-52

        self.dazzlerObject.width = 2.18*self.position**2/(self.c*self.pulsewidth)
        self.dazzlerObject.omega0 = 2*np.pi*self.c/self.position
        self.dazzlerObject.chi0 = self.width/(2*self.position)
        self.dazzlerObject.del_omega0 = self.omega0*(self.chi0-self.chi0**3)
        self.dazzlerObject.omega1 = 2*np.pi*self.c/self.hole_position
        self.dazzlerObject.chi1 = self.hole_width/(2*self.hole_position)
        self.dazzlerObject.del_omega1 = self.omega1*(self.chi1-self.chi1**3)/2
    def run_dazzler(self, input):


        return 
    def initialize_amplifier(self, amplifier_parameters):

    def initialize_sfg(self,)
        self.usingSFG = assertTrue
        u = pyssnl.UNITS()
