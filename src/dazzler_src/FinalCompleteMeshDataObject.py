#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 11:59:01 2021

@author: student
"""

from pypret import lib

def complete_object(trace_mdo, time_vector, freq_vector):
    import pypret
    import numpy as np 
            
    #make this a pnps-type object? so that i can input it for retrieval
    N=len(time_vector)
    dt=time_vector[1]-time_vector[0]
    
    ourspec=trace_mdo #this is named like this because it should already be a MeshDataObject    
    ourspec.ft=pypret.FourierTransform(N, dt=dt)
    #print(ourspec.ft)
    ourspec.process_w = freq_vector
    #other options (basically higher factors) that could be tried

    #all of this just catches our basic 2D array, brought in as a "MeshData" object, up to the having "PNPS" characteristics
    def calculate(self, spectrum, parameter):
            parameter = np.atleast_1d(parameter)
            Tmn = np.zeros((parameter.size, spectrum.size))
            Smk = np.zeros((parameter.size, spectrum.size), dtype=np.complex128)
            for m, p in enumerate(parameter):
                Tmn[m, :], Smk[m, :] = self._calculate(spectrum, p)
            # if a scalar parameter was used, squeeze out one dimension
            Tmn = Tmn.squeeze()
            Smk = Smk.squeeze()
            # store for later use (in self.trace)
            self.Tmn = Tmn
            self.Smk = Smk
            self.parameter = parameter
            self.spectrum = spectrum
            return Tmn
    def _calculate(self, spectrum, parameter):
            """ Calculates the nonlinear process spectrum for a single parameter.
            Follows the notation from our paper.
            """
            ft = self.ft
            delay, Ak, Ek, Sk, Tn = self._get_tmp(parameter)
            ft.backward(delay * spectrum, out=Ak)
            ft.backward(spectrum, out=Ek)
            if self.process == "shg":
                Sk[:] = Ak * Ek
            elif self.process == "pg":
                Sk[:] = lib.abs2(Ak) * Ek
            elif self.process == "tg":
                Sk[:] = Ak.conj() * (Ek * Ek)
            Tn[:] = self.measure(Sk)
            return Tn, Sk
    def _get_tmp(self, parameter):
            if parameter in self._tmp:
                return self._tmp[parameter]
            delay = np.exp(1.0j * parameter * self.ft.w)
            N = self.ft.N
            Ak = np.zeros(N, dtype=np.complex128)
            Ek = np.zeros(N, dtype=np.complex128)
            Sk = np.zeros(N, dtype=np.complex128)
            Tn = np.zeros(N, dtype=np.float64)
            self._tmp[parameter] = delay, Ak, Ek, Sk, Tn
            return delay, Ak, Ek, Sk, Tn
    def measure(self, Sk):
            """ Simulates the measurement process.
            Note that we deal with the spectrum over the frequency!
            For retrieving from actual data we need to rescale this by lambda^2.
            """
            Sn = self.ft.forward(Sk)
            return lib.abs2(Sn)
    def gradient(self, Smk2, parameter):
            """ Calculates the gradient ∇_n Z_m.
            """
            parameter = np.atleast_1d(parameter)
            Smk2 = np.atleast_2d(Smk2)
            gradnZm = np.zeros((parameter.shape[0], Smk2.shape[1]),
                               dtype=np.complex128)
            for m, p in enumerate(parameter.flat):
                gradnZm[m, :] = self._gradient(Smk2[m, :], p)
            # if a scalar parameter is passed, squeeze out one dimension
            return gradnZm.squeeze()

    #may be able to simplify if we know which method to use
    def _gradient(self, Sk2, parameter):
            """ Returns the gradient of Z based on the previous call to _spectrum.
            """
            ft = self.ft
            # retrieve the intermediate results
            delay, Ak, Ek, Sk, Tn = self._tmp[parameter]
            # difference between original and updated PNPS signal
            dSk = Sk2 - Sk
            # calculate the gradients as described in the supplement
            if self.process == "shg":
                gradnZ = (delay.conj() * ft.forward(dSk * Ek.conj()) +
                          ft.forward(dSk * Ak.conj()))
            elif self.process == "pg":
                gradnZ = (2 * delay.conj() *
                          ft.forward(Ak * np.real(dSk * Ek.conj())) +
                          ft.forward(dSk * lib.abs2(Ak)))
            elif self.process == "tg":
                gradnZ = (delay.conj() * ft.forward(dSk.conj() * Ek * Ek) +
                          2 * ft.forward(dSk * Ek.conj() * Ak))
            # common scale for all gradients (note the minus)
            gradnZ *= -2.0 * lib.twopi * ft.dw / ft.dt
            return gradnZ

    ret = pypret.Retriever(ourspec, "copra", verbose=False, maxiter=30)
    ourspec.calculate = calculate.__get__(ourspec)
    ourspec._calculate = _calculate.__get__(ourspec)
    ourspec._get_tmp = _get_tmp.__get__(ourspec)
    ourspec._tmp=dict()
    ourspec.process="pg" #also try out shg and tg, whatever that is?? - the other two supported methods for frog
    ourspec.measure = measure.__get__(ourspec)
    ourspec.gradient = gradient.__get__(ourspec)
    ourspec._gradient = _gradient.__get__(ourspec)

    return ourspec