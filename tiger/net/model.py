from typing import Dict


SPIKE_GENERATOR = "spike_generator"
IAF_COND_ALPHA = "iaf_cond_alpha"
NOISE_GENERATOR = "noise_generator"


class Model:
    _name: str
    _params: Dict
    
    def __init__(self, name: str, params: Dict) -> None:
        self._name = name
        self._params = params


# Ganglion cells in retinas act as spike generators.
# Spikes are given as an array.
def retinal_ganglion_cell() -> Model:
    params = {"origin": 0.0, "start": 0.0}
    return Model(SPIKE_GENERATOR, params)


def lgn_relay_cell() -> Model:
    params = {
        "C_m": 100.0,
        "g_L": 10.0,
        "E_L": -60.0,
        "V_th": -55.0,
        "V_reset": -60.0,
        "t_ref":  2.0,
        "E_ex": 0.0, # AMPA, from Hill-Tononi 2005
        "E_in": -80.0, # GABA-A of thalamocortical cells, from Hill-Tononi 2005
        "tau_syn_ex": 1.0, # it approximates Hill-Tononi's diff. of exp. response, also Casti 2008
        "tau_syn_in": 3.0 # it approximates Hill-Tononi's diff. of exp. response
    }
    return Model(IAF_COND_ALPHA, params)


def lgn_interneuron_cell() -> Model:
    params = {
        "C_m": 100.0,
        "g_L": 10.0,
        "E_L": -60.0,
        "V_th": -55.0,
        "V_reset": -60.0,
        "t_ref":  2.0,
        "E_ex": 0.0,
        "E_in": -80.0,
        "tau_syn_ex": 1.0,
        "tau_syn_in": 3.0
    }
    return Model(IAF_COND_ALPHA, params)


def cortex_excitatory_cell() -> Model:
    params = {
        "C_m": 100.0,
        "g_L": 10.0,
        "E_L": -60.0,
        "V_th": -55.0,
        "V_reset": -60.0,
        "t_ref":  2.0,
        "E_ex": 0.0,
        "E_in": -70.0, # GABA-A of cortical cells, from Hill-Tononi 2005
        "tau_syn_ex": 1.0,
        "tau_syn_in": 3.0
    }
    return Model(IAF_COND_ALPHA, params)


def cortex_inhibitory_cell() -> Model:
    params = {
        "C_m": 100.0,
        "g_L": 10.0,
        "E_L": -60.0,
        "V_th": -55.0,
        "V_reset": -60.0,
        "t_ref":  2.0,
        "E_ex": 0.0,
        "E_in": -70.0,
        "tau_syn_ex": 1.0,
        "tau_syn_in": 3.0
    }
    return Model(IAF_COND_ALPHA, params)


# A Gaussian noise generator
def thalamo_noise() -> Model:
    params = {'mean': 0.0, 'std': 1.0}
    return Model(NOISE_GENERATOR, params)
