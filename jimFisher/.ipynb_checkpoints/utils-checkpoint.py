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
DEFAULT_FISHER_PARAMS = ['log_chirp_mass', 'mass_ratio', 's1_z', 's2_z', 'snr', 'cos_iota']
WF_FUNCS = {"IMRPhenomD": jimgw.single_event.waveform.RippleIMRPhenomD}

BOUNDS = {'mass_ratio': [0,1],
         'cos_iota': [-1, 1],
         's1_z': [-1, 1],
          's2_z': [-1,1],
          'snr': [0, np.inf]
         }
LABELS = {'chirp_mass': r"$\mathcal{M}_c$",
         'mass_ratio': r"$q$",
         'luminosity_distance': r"$D_L$",
         'dec': r"$\delta$",
         'ra': r"$\alpha$",
         's1_z': r"$s_{1z}$",
          's2_z': r"$s_{2z}$",
          'cos_iota': r"$\cos\iota$",
         'snr': r"$\rho$",
         'mass_1': r"$m_1$",
         'mass_2': r"$m_2$"}

mass_params=['mass_1', 'mass_2', 'component_masses', 'total_mass', 'symmetric_mass_ratio']

def convert_to_ripple_params(params):
    converted = params.copy()
    
    converted["eta"] = params['mass_ratio'] / (1 + params['mass_ratio']) ** 2
    converted['M_c'] = jnp.exp(params['log_chirp_mass'])
    converted['iota'] = jnp.arccos(params['cos_iota'])
    converted['dec'] = jnp.arcsin(params['sin_dec'])
    converted['d_L'] = jnp.exp(params['log_luminosity_distance'])
    
    return converted

def convert_to_fisher_params(params, fisher_params):
    converted = params.copy()
    for fisherkey in fisher_params:
        if fisherkey.startswith('log_'):
            converted[fisherkey] = jnp.log(params[fisherkey.split("log_")[1]])
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

def convert_to_physical(samples, rho0=None, dL0=None, generate=['component_masses', 'chi_eff', 'luminosity_distance']):
    keys = list(samples.keys())
    converted = samples.copy()
    for key in samples.keys():
        if key.startswith('log_'):
            converted[key[4:]] = jnp.exp(samples[key])
            keys.remove(key)
            keys.append(key[4:])
    if ('snr' in keys) and ('luminosity_distance' not in keys):
        if rho0 is not None:
            converted['luminosity_distance'] = rho0 * dL0 / converted['snr']
            keys.append('luminosity_distance')
    if any(mass_param in generate for mass_param in mass_params):
        converted_masses = dict()
        converted_masses['symmetric_mass_ratio'] = converted['mass_ratio'] / (1 + converted['mass_ratio'])**2
        converted_masses['total_mass'] = converted_masses['symmetric_mass_ratio']**(-3/5)
        converted_masses['mass_1'] = converted_masses['total_mass'] / (1 + converted['mass_ratio'])
        converted_masses['mass_2'] = converted_masses['mass_1'] * converted['mass_ratio']
        masskeys = [m for m in mass_params if m in generate]
        for key in masskeys:
            if 'component_mass' in key:
                converted['mass_1'] = converted_masses['mass_1']
                converted['mass_2'] = converted_masses['mass_2']
                keys.extend(['mass_1', 'mass_2'])
            else:
                converted[key] = converted_masses[key]
                keys.append(key)
        
    return {key: converted[key] for key in keys}

def filter_samples(samples):
    keep = jnp.ones_like(samples[list(samples.keys())[0]], dtype=bool)
    for key in BOUNDS.keys():
        keep &= (samples[key] >= BOUNDS[key][0]) & (samples[key] <= BOUNDS[key][-1])
    return {key: samples[key][keep] for key in samples}

def compute_distance(v1, v2, g):
    delta = v2 - v1
    distance_squared = jnp.dot(delta.T, jnp.dot(g, delta))
    distance = jnp.sqrt(distance_squared)
    return distance