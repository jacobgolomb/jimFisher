#!/usr/bin/env python
# coding: utf-8

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
from .utils import filter_samples, convert_to_physical, convert_to_ripple_params, IFO_GEOMETRIC_KEYS, DEFAULT_FISHER_PARAMS, WF_FUNCS, BOUNDS, ASD_FILES, optimal_snr, draw_samples, jimGW_detectors, inner, compute_distance, convert_to_fisher_params, bound_theta

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
        self.true_parameters = convert_to_fisher_params(parameters, self.fisher_parameters)
        self.true_parameters.update(convert_to_ripple_params(parameters))
        self.true_parameters.update({"gmst": self.gmst, "epoch": self.epoch})

        self.theta_vec_true = jnp.array([self.true_parameters[key] for key in self.fisher_parameters])
        
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
        argnames = list(inspect.signature(self._get_signal_detector).parameters.keys())
        fisher_argnums = [ii for ii, param in enumerate(argnames) if param in self.fisher_parameters]
        self.argnames = argnames
        self.fisher_argnums = fisher_argnums
        self.fixed_argnames = [arg for ii, arg in enumerate(argnames) if ii not in fisher_argnums]
    
    #@partial(jax.jit, static_argnums=(0,))    
    def _get_signal_detector(self, log_chirp_mass, logit_mass_ratio, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z, log_luminosity_distance, phase_c, cos_iota,
                            ra, dec, psi, t_c):
        
        params = {"log_chirp_mass": log_chirp_mass, "logit_mass_ratio": logit_mass_ratio, "s1_z": s1_z, "s2_z": s2_z,
        "log_luminosity_distance": log_luminosity_distance, "phase_c": phase_c, "cos_iota":cos_iota, "ra": ra, "dec": dec, 
        "psi": psi, "t_c": t_c, "gmst": self.gmst, "epoch": self.epoch}
        
        converted_parameters = convert_to_ripple_params(params)
        h = self.get_signal_in_detector(converted_parameters)
        return h
    
    #@partial(jax.jit, static_argnums=(0,))
    def _get_scaled_signal_detector(self, log_chirp_mass, mass_ratio, s1_x, s1_y, s1_z, s2_x, s2_y, s2_z, snr, phase_c, cos_iota,
                            ra, dec, psi, t_c, rho0, dL0):
        logdL0 = jnp.log(dL0)
        params = {"log_chirp_mass": log_chirp_mass, "mass_ratio": mass_ratio, "s1_z": s1_z, "s2_z": s2_z,
        "log_luminosity_distance": logdL0, "phase_c": phase_c, "cos_iota": cos_iota, "ra": ra, "dec": dec, 
        "psi": psi, "t_c": t_c, "gmst": self.gmst, "epoch": self.epoch}
        
        converted_parameters = convert_to_ripple_params(params)
        
        h = self.get_signal_in_detector(converted_parameters) * snr / rho0
        return h
    
    #@partial(jax.jit, static_argnums=(0,))
    def get_snr(self, parameters):

        converted_parameters = convert_to_ripple_params(parameters)
        converted_parameters['gmst'] = parameters.get('gmst', self.gmst)
        converted_parameters['epoch'] = self.epoch
        
        signal = self.get_signal_in_detector(converted_parameters)
        snr = optimal_snr(signal, self.ifo.psd, self.frequency_array)
        return snr
    
    #@partial(jax.jit, static_argnums=(0,))
    def get_jacobian(self, parameters):
        
        inputs = self.get_grad_input_args(parameters)
        
        jac = jnp.array(
            jax.jacfwd(self._get_signal_detector, argnums=self.fisher_argnums)(
            *inputs)
        )
        return jac
    @partial(jax.jit, static_argnums=(0,))
    def get_fisher(self, parameters=None):
        jacobian = self.get_jacobian(parameters)
        fisher_matrix = jnp.zeros((len(jacobian), len(jacobian)))
        for ii in range(len(jacobian)):
            for jj in range(ii, len(jacobian)):
                element = inner(jacobian[ii], jacobian[jj], self.ifo.psd, self.frequency_array)
                fisher_matrix = fisher_matrix.at[ii,jj].set(element)
                fisher_matrix = fisher_matrix.at[jj,ii].set(element)    
        return fisher_matrix
    
    #@partial(jax.jit, static_argnums=(0,))
    def get_covariance(self, parameters=None):
        fisher = self.get_fisher(parameters)
        cov = jnp.linalg.inv(fisher)
        return cov
    
    def theta_vec_to_param_dict(self, theta_vec, theta_params_true):
        theta_parameters = {key: theta_vec[ii] for ii, key in enumerate(self.fisher_parameters)}
        for key in self.fixed_argnames:
            theta_parameters[key] = theta_params_true[key]
        return theta_parameters
    
    #@partial(jax.jit, static_argnums=(0,))    
    def update_theta_vec(self, theta_vec, random_normal, theta_vec_true, theta_params_true):
        theta_params = self.theta_vec_to_param_dict(theta_vec, theta_params_true)
        theta_params_clipped = bound_theta(theta_params)
        fisher = self.get_fisher(theta_params_clipped)
        cholesky_fisher = jnp.linalg.cholesky(fisher)
        theta_vec_new = theta_vec_true + jax.scipy.linalg.solve_triangular(cholesky_fisher.T, random_normal, lower=False)
        
        return theta_vec_new, fisher

    def distance_condition(self, theta_vec, theta_vec_old, fisher, threshold):
        distance = compute_distance(theta_vec, theta_vec_old, fisher)
        return distance > threshold

    def iterate_theta_vec(self, theta_vec_true, random_normal, threshold, max_iterations):
        theta_params_true = self.true_parameters
        theta_vec, fisher = self.update_theta_vec(theta_vec_true, random_normal, theta_vec_true, theta_params_true)
        for _ in trange(max_iterations):
            theta_vec_new, fisher = self.update_theta_vec(theta_vec, random_normal, theta_vec_true, theta_params_true)
            if jnp.any(jnp.isnan(theta_vec_new)):
                theta_params = self.theta_vec_to_param_dict(theta_vec, theta_params_true)
                fish = self.get_fisher(theta_params)
                cov = self.get_covariance(theta_params)
                break
            distance = compute_distance(theta_vec, theta_vec_new, fisher)
            theta_vec = theta_vec_new
            if distance < threshold:
                break
        print(distance)

        return theta_vec_new

    
    def iterate_theta_vec_inv(self, theta_vec_true, random_normal, threshold):
        theta_params = self.theta_vec_to_param_dict(theta_vec_true)
        fisher = self.get_fisher(theta_params)
        cov = jnp.linalg.inv(fisher)
        cholesky_cov = jnp.linalg.cholesky(cov)

        theta_vec_cov = theta_vec_true + jnp.matmul(cholesky_cov, random_normal)
        theta_vec = theta_vec_cov  
        for _ in trange(100):
            theta_params = self.theta_vec_to_param_dict(theta_vec)
            fisher = self.get_fisher(theta_params)
            cov = jnp.linalg.inv(fisher)
            cholesky = jnp.linalg.cholesky(cov)
            theta_vec_new = theta_vec_true + jnp.matmul(cholesky, random_normal)
            distance = compute_distance(theta_vec, theta_vec_new, fisher)
            theta_vec = theta_vec_new
            if distance < threshold:
                break
        return theta_vec_new
    
    def set_observed_means(self, seed=None, zero_noise=False):

        
        fisher = self.get_fisher(self.true_parameters)
        fisher_cholesky = jnp.linalg.cholesky(fisher)
        
        if not zero_noise:
            if seed is None:
                seed = np.random.randint(0, 10000)
            key = jax.random.PRNGKey(seed)
            random_normal = jax.random.normal(key, shape=(len(self.fisher_parameters),))
            theta_vec_observed = self.theta_vec_true + jax.scipy.linalg.solve_triangular(fisher_cholesky.T, random_normal, lower=False)
        else:
            theta_vec_observed = self.theta_vec_true
        self.observed_means = theta_vec_observed
        self.observed_parameters = self.theta_vec_to_param_dict(theta_vec_observed, self.true_parameters)
        self.observed_fisher = fisher

    
    def set_observed_means_iterate(self, seed=None, max_iterations = 100, threshold=1e-11):
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        random_normal = jax.random.normal(key, shape=(len(self.fisher_parameters),))
        theta_vec_final=self.iterate_theta_vec(self.theta_vec_true, random_normal, threshold, max_iterations)

        theta_params_final = self.theta_vec_to_param_dict(theta_vec_final, self.true_parameters)
        theta_params_final_clipped= bound_theta(theta_params_final)
        self.observed_means = theta_vec_final
        self.observed_parameters_clipped = theta_params_final_clipped
        self.observed_parameters = theta_params_final
        
    def set_observed_means_inv(self, seed=None, N=1000, max_iterations = 100, threshold=1e-11):
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.PRNGKey(seed)
        random_normal = jax.random.normal(key, shape=(len(self.fisher_parameters),))
        self.theta_vec_true = jnp.array([self.true_parameters[key] for key in self.fisher_parameters])
        theta_vec_final=self.iterate_theta_vec_inv(self.theta_vec_true, random_normal, threshold)

        theta_params_final = self.theta_vec_to_param_dict(theta_vec_final, self.theta_vec_true)
        theta_vec_final_clipped= bound_theta(theta_params_final)
        theta_params_final_clipped = self.theta_vec_to_param_dict(theta_vec_final_clipped, self.theta_vec_true)
        self.observed_means = theta_vec_final
        self.observed_parameters_clipped = theta_params_final_clipped
        self.observed_parameters = theta_params_final
        
    def set_observed_fisher(self):
        self.observed_fisher = self.get_fisher(self.true_parameters)
    
    def set_observed_properties(self, seed=None, N=1000, zero_noise=False):
        self.set_observed_means(seed=seed, zero_noise=zero_noise)
        self.set_observed_fisher()
        self.observed_snr = self.get_snr(self.observed_parameters)
        
    def set_observed_properties_inv(self, seed=None, N=1000, max_iterations = 100, threshold=1e-11):
        self.set_observed_means_inv(seed=seed, N=N, max_iterations=max_iterations, threshold=threshold)
        self.set_observed_fisher()
        
    def draw_observed_samples(self, seed=None, N=1000):
        if seed is None:
            seed = np.random.randint(0, 10000)
        key = jax.random.key(seed)
        
        observed_samples = {param: [] for param in self.fisher_parameters}
        print("drawing observed samples")
        
        while len(observed_samples[list(observed_samples.keys())[0]]) < N:
            samples = draw_samples(mu=self.observed_means, fisher=self.observed_fisher, 
                                   names=self.fisher_parameters,key=key, N=N)
            samples = filter_samples(samples)
            for parameter in observed_samples.keys():
                observed_samples[parameter].extend(samples[parameter])
                
        return {parameter: jnp.array(observed_samples[parameter][:N]) for parameter in observed_samples}
            
    def draw_physical_samples(self, seed=None, N=1000, generate=['component_masses', 'chi_eff', 'luminosity_distance', 'cos_iota']):
        
        original_samples = self.draw_observed_samples(seed=seed, N=N)
        original_samples.update({key: self.true_parameters[key] * jnp.ones(N) for key in self.fixed_argnames})
        
        physical_samples = convert_to_physical(original_samples, generate=generate)
        return physical_samples
    