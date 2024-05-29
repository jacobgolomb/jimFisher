import jimgw
import jax
import jax.numpy as jnp
import os
import numpy as np

sensitivity_path = f"{os.path.dirname(__file__)}/sensitivity_files"
jimGW_detectors = jimgw.single_event.detector.detector_preset

ASD_FILES = {"CE": f"{sensitivity_path}/LIGO-P1600143-v18-CE-ASD.txt",
            "aplus": f"{sensitivity_path}/AplusDesign.txt",
            "aligo": f"{sensitivity_path}/aligo_O4high.txt"}

IFO_GEOMETRIC_KEYS = ['latitude', 'longitude',
                    'elevation', 'xarm_azimuth', 'yarm_azimuth',
                    'xarm_tilt', 'yarm_tilt']
DEFAULT_FISHER_PARAMS = ['logM_c', 'q', 's1_z', 's2_z', 'rho',  'cos_iota']
WF_FUNCS = {"IMRPhenomD": jimgw.single_event.waveform.RippleIMRPhenomD}

BOUNDS = {'q': [0,1],
         'cos_iota': [-1, 1],
         's1_z': [-1, 1],
          's2_z': [-1,1],
          'rho': [0, np.inf]
         }

def convert_to_ripple_params(params):
    converted = params.copy()
    
    converted["eta"] = params['q'] / (1 + params['q']) ** 2
    converted['M_c'] = jnp.exp(params['logM_c'])
    converted['iota'] = jnp.arccos(params['cos_iota'])
    converted['dec'] = jnp.arcsin(params['sin_dec'])
    converted['d_L'] = jnp.exp(params['logdL'])
    
    return converted

def inner(a, b, psd_array, freqs):
    integrand = (jnp.conjugate(a) * b ) / psd_array
    return 4 * jnp.real( jnp.trapezoid(y=integrand, x=freqs, axis=-1))
    
def optimal_snr(signal, psd, freqs):
    return jnp.sqrt(inner(signal, signal, psd, freqs))

def draw_samples(mu, covariance, names, key, N=1000):
    samples = jax.random.multivariate_normal(key = key, 
                                                   mean=mu, 
                                                   cov=covariance, shape=(N,))
    samples = {names[i]: samples[:,i] for i in range(len(names))}   
    return samples

def convert_to_physical(samples, rho0=None, dL0=None):
    keys = list(samples.keys())
    converted = samples.copy()
    for key in samples.keys():
        if key.startswith('log'):
            converted[key[3:]] = jnp.exp(samples[key])
            keys.remove(key)
            keys.append(key[3:])
    if ('rho' in keys) and ('d_L' not in keys):
        if rho0 is not None:
            converted['d_L'] = rho0 * dL0 / converted['rho']
            keys.append('d_L')
    return {key: converted[key] for key in keys}

def filter_samples(samples):
    keep = jnp.ones_like(samples[list(samples.keys())[0]], dtype=bool)
    for key in BOUNDS.keys():
        keep &= (samples[key] >= BOUNDS[key][0]) & (samples[key] <= BOUNDS[key][-1])
    return {key: samples[key][keep] for key in samples}

