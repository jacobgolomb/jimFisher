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
DEFAULT_FISHER_PARAMS = ['log_chirp_mass', 'mass_ratio', 's1_z', 's2_z', 'log_luminosity_distance', 'cos_iota']
WF_FUNCS = {"IMRPhenomD": jimgw.single_event.waveform.RippleIMRPhenomD,
           "IMRPhenomPv2": jimgw.single_event.waveform.RippleIMRPhenomPv2}

BOUNDS = {'mass_ratio': [0,1],
         'cos_iota': [-1, 1],
         's1_z': [-1, 1],
          's2_z': [-1,1],
          'snr': [0, np.inf],
          'luminosity_distance': [0, 100000]

         }

def bound_theta(parameters):
    for bounded_param in BOUNDS:
        for prefix in inverse_operations:
            if prefix+bounded_param in parameters:
                bounded = jnp.clip(inverse_operations[prefix](parameters.pop(prefix+bounded_param)),
                                                     BOUNDS[bounded_param][0] + 1e-6, BOUNDS[bounded_param][1] - 1e-6
                                  )
                parameters[prefix+bounded_param] = operations[prefix](bounded)
    return parameters

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
         'mass_2': r"$m_2$",
         'chi_eff': r"$\chi_{\rm eff}$"}

K = 35
inverse_operations = {"log_": jnp.exp,
                     "sin_": jnp.arcsin,
                     "cos_": lambda x: jnp.arccos(jnp.clip(x, -1.0 + 1e-6, 1.0 - 1e-6)),
                      "logit_": jax.scipy.special.expit,
             "onelogit_":lambda f: jnp.nan_to_num((jnp.exp(f/K) - 1)/(1+jnp.exp(f/K)), nan=jnp.inf)
                     }

operations = {"log_": jnp.log,
                     "sin_": jnp.sin,
                     "cos_": jnp.cos,
             "logit_": jax.scipy.special.logit,
             "onelogit_": lambda x: K *(jnp.log(1+x) - jnp.log(1-x))}

RIPPLE_KEYS = ['M_c', 'eta', 'd_L', 'iota', 'ra', 'dec', 'phase_c', 'psi', 's1_x', 's1_y', 's1_z', 's2_x', 's2_y', 's2_z', 't_c', 'gmst', 'epoch']
RIPPLE_MAP = {'M_c': 'chirp_mass', 'd_L': 'luminosity_distance'} 
    
mass_params=['mass_1', 'mass_2', 'component_masses', 'total_mass', 'symmetric_mass_ratio']

def convert_to_ripple_params(params):
    converted = dict()
    temp = params.copy()
    for parameter in params:
        for prefix in inverse_operations:
            if parameter.startswith(prefix):
                untransformed_param = parameter.split(prefix)[1]
                temp[untransformed_param] = inverse_operations[prefix](params[parameter])
                for pref in inverse_operations:
                    if untransformed_param.startswith(pref):
                        temp[untransformed_param.split(pref)[1]] = inverse_operations[pref](temp[untransformed_param])

    temp["eta"] = temp['mass_ratio'] / (1 + temp['mass_ratio']) ** 2

    for parameter in RIPPLE_KEYS:
        original_key = RIPPLE_MAP.get(parameter, parameter)
        if original_key in temp: 
            converted[parameter] = temp[original_key]
    
    return converted

def convert_to_fisher_params(params, fisher_params):
    converted = dict()
    
    for fisherkey in fisher_params:
        for prefix in operations:
            if fisherkey.startswith(prefix):
                regular_key = fisherkey.split(prefix)[1]
                if regular_key in params:
                    converted[prefix+regular_key] = operations[prefix](params[regular_key])
        if fisherkey not in converted:
            converted[fisherkey] = params[fisherkey]
    return converted

def inner(a, b, psd_array, freqs):
    integrand = (jnp.conjugate(a) * b ) / psd_array
    return 4 * jnp.real( jnp.trapezoid(y=integrand, x=freqs, axis=-1))
    
def optimal_snr(signal, psd, freqs):
    return jnp.sqrt(inner(signal, signal, psd, freqs))

def draw_samples(mu, fisher, names, key, N=1000):
    samples_unit = jax.random.multivariate_normal(key = key, 
                                                   mean=jnp.zeros(mu.shape), cov=jnp.eye(len(mu)),shape=(N,))
    chol = jnp.linalg.cholesky(fisher)
    scaled_samples = jax.scipy.linalg.solve_triangular(chol.T, samples_unit.T, lower=False).T

    samples = mu + scaled_samples
    samples = {names[i]: samples[:,i] for i in range(len(names))}   
    return samples

def convert_to_physical(samples, rho0=None, dL0=None, generate=['component_masses', 'chi_eff', 'luminosity_distance', 'cos_iota']):
    if generate is None:
        generate=[]
    keys = list(samples.keys())
    converted = samples.copy()
    for prefix in inverse_operations:
        for key in samples:
            if key.startswith(prefix):
                newkey = key.split(prefix)[1]
                converted[newkey] = inverse_operations[prefix](samples[key])
                keys.remove(key)
                keys.append(newkey)
    if ('snr' in keys) and ('luminosity_distance' not in keys):
        if rho0 is not None:
            converted['luminosity_distance'] = rho0 * dL0 / converted['snr']
            keys.append('luminosity_distance')
    if any(mass_param in generate for mass_param in mass_params):
        converted_masses = dict()
        converted_masses['symmetric_mass_ratio'] = converted['mass_ratio'] / (1 + converted['mass_ratio'])**2
        converted_masses['total_mass'] = converted_masses['symmetric_mass_ratio']**(-3/5) * converted['chirp_mass']
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
    if 'chi_eff' in generate:
        converted['chi_eff'] = (converted['s1_z'] + converted['mass_ratio'] * converted['s2_z']) / (1 + converted['mass_ratio'])
        keys.append('chi_eff')
    for parameter in generate:
        for operation in operations:
            if (parameter.startswith(operation)):
                if parameter in converted:
                    keys.append(parameter)
                else:
                    converted[parameter] = operations[operation](converted[parameter.split(operation)[1]])
                    keys.append(parameter)
        
    return {key: converted[key] for key in keys}

def filter_samples(samples):
    samples_converted = convert_to_physical(samples,generate=None)
    keep = jnp.ones_like(samples[list(samples.keys())[0]], dtype=bool)
    for key in samples_converted.keys():
        if key in BOUNDS:
            keep &= (samples_converted[key] >= BOUNDS[key][0]) & (samples_converted[key] <= BOUNDS[key][-1])
    return {key: samples[key][keep] for key in samples}

def compute_distance(v1, v2, g):
    delta = v2 - v1
    distance_squared = jnp.dot(delta.T, jnp.dot(g, delta))
    distance = jnp.sqrt(distance_squared)
    return distance