#!/usr/bin/env python
# coding: utf-8

# In[294]:

import jimgw
from functools import partial
from astropy.time import Time
from scipy.interpolate import interp1d
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
import tqdm
import inspect
import jimgw.single_event.detector
import jimgw.single_event.waveform
from tqdm import trange
from .utils import filter_samples, convert_to_physical, convert_to_ripple_params, IFO_GEOMETRIC_KEYS, DEFAULT_FISHER_PARAMS, WF_FUNCS, BOUNDS, ASD_FILES, optimal_snr, draw_samples, jimGW_detectors, inner, compute_distance, convert_to_fisher_params

class FisherSamples(object):
    def __init__(self, name="CE", fmin = 20, fmax = None, 
                 sampling_frequency = None, asd_file=None, psd_file=None, sensitivity=None, location=None, duration=None,
                 trigger_time=None, post_trigger_duration=2,
                waveform="IMRPhenomD", f_ref=None, fisher_parameters=None):
        self.name = name
        self.set_sensitivity_file(psd_file, asd_file, sensitivity)
        self.set_ifo_geometry(location)
        
        self.set_times(trigger_time, duration, post_trigger_duration)
        self.set_frequencies(fmin, fmax, sampling_frequency)
        self.setup_ifo_object()
        
        if f_ref is None:
            self.f_ref=20.0
        else:
            self.f_ref = float(f_ref)
        self.wf_func = WF_FUNCS[waveform](f_ref=self.f_ref)
        
        if fisher_parameters is None:
            self.fisher_parameters = DEFAULT_FISHER_PARAMS
        else:
            self.fisher_parameters = fisher_parameters
        self.set_argument_orders()
            
    def set_times(self,trigger_time, duration, post_trigger):
        epoch = duration - post_trigger
        gmst = Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad #gmst of the trigger time in the detector
        self.epoch = epoch
        self.gmst = gmst
        self.duration = duration
        
    def set_at_true(self, parameters):
        self.true_parameters = convert_to_fisher_params(parameters, self.fisher_parameters + ["log_luminosity_distance"])
        self.true_parameters.update(convert_to_ripple_params(self.true_parameters))
        self.true_parameters.update({"gmst": self.gmst, "epoch": self.epoch})
        snr = self.get_snr(self.true_parameters)
        self.true_parameters['rho0'] = snr
        self.true_parameters['snr'] = snr
        self.true_parameters['dL0'] = self.true_parameters['d_L']
        self.covariance_at_true = self.get_covariance(self.true_parameters)
        
    def set_ifo_geometry(self, location):
        if location is None:
            location = self.name
        self.location = location
        self.geometry = {key: jimGW_detectors[location].__dict__[key] for key in IFO_GEOMETRIC_KEYS}
            
    def set_sensitivity_file(self, psd_file, asd_file, sensitivity):
        if psd_file:
            self.sensitivity_file = psd_file
            self.sensitivity_type="PSD"
        elif asd_file:
            self.sensitivity_file = asd_file
            self.sensitivity_type = "ASD"
        else:
            if sensitivity is not None:
                ifo_sensitivity = sensitivity
            else:
                raise ValueError("Must specify sensitivity if not providing a sensitivity file")
            self.sensitivity_file = ASD_FILES[ifo_sensitivity]
            self.sensitivity_type = "ASD"

    def set_frequencies(self, fmin, fmax, srate):
        self.minimum_frequency = fmin
        if not fmax:
            self.sampling_frequency = srate
            self.maximum_frequency = self.sampling_frequency / 2
        elif not srate:
            self.maximum_frequency = fmax
            self.sampling_frequency = 2 * self.maximum_frequency
        else:
            self.maximum_frequency = fmax
            self.sampling_frequency = srate
        Nfft = self.duration * self.sampling_frequency
        frequency_array = np.linspace(0, self.maximum_frequency, Nfft, endpoint=False)
        frequency_array = jnp.array(frequency_array[(frequency_array >= self.minimum_frequency) 
                                                    & (frequency_array <= self.maximum_frequency)])
        self.frequency_array = frequency_array
    
    def get_safe_duration(self, parameters):
        signal_duration = calculate_time_to_merger(self.minimum_frequency, parameters['mass_1'], parameter['mass_2'])
        safe_duration = max(np.ceil(signal_duration), 4)
        return safe_duration
    
    def setup_ifo_object(self):

        ifo = jimgw.detector.GroundBased2G(name=self.name,
                       **self.geometry)

        # Placing CE in Hanford so we take the geometric/location parameters of H1 and just replace the PSD with CE's PSD

        ifo.psd = self.get_ifo_psd()
        self.ifo = ifo
    
    def get_ifo_psd(self):
        
        freqs, sens = np.loadtxt(self.sensitivity_file, unpack=True)
        
        if self.sensitivity_type == "ASD":

            psd = sens**2
        else:
            psd = sens

        psd_interped = interp1d(freqs, psd, fill_value=(psd[0], psd[-1]))(self.frequency_array)
        return jnp.array(psd_interped)
    
    def get_signal_in_detector(self, parameters):
        h_sky = self.wf_func(self.frequency_array, parameters)
        align_time = jnp.exp(
                -1j * 2 * jnp.pi * self.frequency_array * (self.epoch + parameters["t_c"])
            )
        return self.ifo.fd_response(self.frequency_array, h_sky, parameters) * align_time
    
    def get_grad_input_args(self, params):
        inputs = []
        for param in self.argnames:
            if param in params:
                inputs.append(params[param])
            else:
                inputs.append(self.__dict__[param])
        return inputs
    
    def set_argument_orders(self):
        argnames = list(inspect.signature(self._get_scaled_signal_detector).parameters.keys())
        fisher_argnums = [ii for ii, param in enumerate(argnames) if param in self.fisher_parameters]
        self.argnames = argnames
        self.fisher_argnums = fisher_argnums
        self.fixed_argnames = [arg for ii, arg in enumerate(argnames) if ii not in fisher_argnums]
    
    def _get_signal_detector(self, log_chirp_mass, mass_ratio, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z, logdL, phase_c, cos_iota,
                            ra, sin_dec, psi, t_c):
        
        params = {"log_chirp_mass": log_chirp_mass, "mass_ratio": mass_ratio, "s1_z": s1_z, "s2_z": s2_z,
        "log_luminosity_distance": log_luminosity_distance, "phase_c": phase_c, "cos_iota": cos_iota, "ra": ra, "sin_dec": sin_dec, 
        "psi": psi, "t_c": t_c, "gmst": self.gmst, "epoch": self.epoch}
        
        converted_parameters = convert_to_ripple_params(params)
        
        h = self.get_signal_in_detector(converted_parameters)
        return h
    
    @partial(jax.jit, static_argnums=(0,))
    def _get_scaled_signal_detector(self, log_chirp_mass, mass_ratio, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z, snr, phase_c, cos_iota,
                            ra, sin_dec, psi, t_c, rho0, dL0):
        logdL0 = jnp.log(dL0)
        params = {"log_chirp_mass": log_chirp_mass, "mass_ratio": mass_ratio, "s1_z": s1_z, "s2_z": s2_z,
        "log_luminosity_distance": logdL0, "phase_c": phase_c, "cos_iota": cos_iota, "ra": ra, "sin_dec": sin_dec, 
        "psi": psi, "t_c": t_c, "gmst": self.gmst, "epoch": self.epoch}
        
        converted_parameters = convert_to_ripple_params(params)
        
        h = self.get_signal_in_detector(converted_parameters) * snr / rho0
        return h
    
    @partial(jax.jit, static_argnums=(0,))
    def get_snr(self, parameters):
        fisher_params = convert_to_fisher_params(parameters, self.fisher_parameters + ["log_luminosity_distance"])
        converted_parameters = convert_to_ripple_params(fisher_params)

        converted_parameters['gmst'] = parameters.get('gmst', self.gmst)
        converted_parameters['epoch'] = self.epoch
        
        signal = self.get_signal_in_detector(converted_parameters)
        snr = optimal_snr(signal, self.ifo.psd, self.frequency_array)
        return snr

    def get_jacobian(self, parameters = None):
        if not parameters:
            parameters = self.parameters
        
        inputs = self.get_grad_input_args(parameters)
        
        jac = jnp.array(
            jax.jacfwd(self._get_scaled_signal_detector, argnums=self.fisher_argnums)(
            *inputs)
        )
        return jac
    
    def get_fisher(self, parameters=None):
        jacobian = self.get_jacobian(parameters)
        fisher_matrix = jnp.zeros((len(jacobian), len(jacobian)))
        for ii in range(len(jacobian)):
            for jj in range(ii, len(jacobian)):
                element = inner(jacobian[ii], jacobian[jj], self.ifo.psd, self.frequency_array)
                fisher_matrix = fisher_matrix.at[ii,jj].set(element)
                fisher_matrix = fisher_matrix.at[jj,ii].set(element)    
        return fisher_matrix
    
    @partial(jax.jit, static_argnums=(0,))
    def get_covariance(self, parameters=None):
        fisher = self.get_fisher(parameters)
        cov = jnp.linalg.inv(fisher)
        return cov
    
    def theta_vec_to_param_dict(self, theta_vec):
        theta_parameters = {key: theta_vec[ii] for ii, key in enumerate(self.fisher_parameters)}
        for key in self.fixed_argnames:
            theta_parameters[key] = self.true_parameters[key]
        return theta_parameters
            
    
    def set_observed_properties(self, seed=None, N=1000, max_iterations = 100, threshold=1e-12):
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.key(seed)
        random_normal = np.random.normal(size=len(self.fisher_parameters))
        cholesky = jnp.linalg.cholesky(self.covariance_at_true)
        theta_vec_true = jnp.array([self.true_parameters[key] for key in self.fisher_parameters])
        theta_vec = theta_vec_true + jnp.matmul(cholesky, random_normal)
        theta_params = self.theta_vec_to_param_dict(theta_vec)
        self.theta_vectors = []
        
        for it in trange(max_iterations):
            cov = self.get_covariance(theta_params)
            cholesky = jnp.linalg.cholesky(cov)
            theta_vec_new = theta_vec_true + jnp.matmul(cholesky, random_normal)
            theta_params_new = self.theta_vec_to_param_dict(theta_vec_new)
            self.theta_vectors.append(np.array(theta_vec_new))
            distance = compute_distance(theta_vec, theta_vec_new, jnp.linalg.inv(cov))
            if distance < threshold:
                print(f"Converged after {it} iterations")
                break
            theta_vec = theta_vec_new
            theta_params = theta_params_new
            
        
        self.observed_covariance = self.get_covariance(theta_params)
        self.observed_means = theta_vec
        
    def draw_observed_samples(self, seed=None, N=1000):
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.key(seed)
        
        observed_samples = {param: [] for param in self.fisher_parameters}
        
        while len(observed_samples[list(observed_samples.keys())[0]]) < N:
            samples = draw_samples(mu=self.observed_means, covariance=self.observed_covariance, 
                                   names=self.fisher_parameters,key=key, N=N)
            samples = filter_samples(samples)
            for parameter in observed_samples.keys():
                observed_samples[parameter].extend(samples[parameter])
                
        return {parameter: jnp.array(observed_samples[parameter][:N]) for parameter in observed_samples}
            
    def draw_physical_samples(self, seed=None, N=1000):
        
        original_samples = self.draw_observed_samples(seed=seed, N=N)
        
        physical_samples = convert_to_physical(original_samples, dL0=self.true_parameters['dL0'], rho0=self.true_parameters['rho0'])
        return physical_samples
    