from typing import Dict, Tuple, List


SPIKE_GENERATOR = "spike_generator"
IAF_COND_ALPHA = "iaf_cond_alpha"
NOISE_GENERATOR = "noise_generator"

RETINAL_GANGLION_CELL = "Retinal_ganglion_cell"
LGN_RELAY_CELL = "LGN_relay_cell"
LGN_INTERNEURON = "LGN_interneuron"
CORTEX_EXC_CELL = "Cortex_excitatory_cell"
CORTEX_INH_CELL = "Cortex_inhibitory_cell"
THALAMO_NOISE = "thalamo_noise"
SYN = "syn"
STATIC_SYNAPSE = "static_synapse"


class Model:
    name: str
    params: Dict
    
    def __init__(self, name: str, params: Dict) -> None:
        self.name = name
        self.params = params


def get_models() -> List[Tuple[str, str, Dict]]:
    retinal_ganglion_cell = _retinal_ganglion_cell()
    lgn_relay_cell = _lgn_relay_cell()
    lgn_interneuron_cell = _lgn_interneuron_cell()
    cortex_exc_cell = _cortex_excitatory_cell()
    cortex_inh_cell = _cortex_inhibitory_cell()
    thalamo_noise = _thalamo_noise()
    
    return [
        (retinal_ganglion_cell.name, RETINAL_GANGLION_CELL, retinal_ganglion_cell.params),
        (lgn_relay_cell.name, LGN_RELAY_CELL, lgn_relay_cell.params),
        (lgn_interneuron_cell.name, LGN_INTERNEURON, lgn_interneuron_cell.params),
        (cortex_exc_cell.name, CORTEX_EXC_CELL, cortex_exc_cell.params),
        (cortex_inh_cell.name, CORTEX_INH_CELL, cortex_inh_cell.params),
        (thalamo_noise.name, THALAMO_NOISE, thalamo_noise.params),
    ]


# Ganglion cells in retinas act as spike generators.
# Spikes are given as an array.
def _retinal_ganglion_cell() -> Model:
    params = {"origin": 0.0, "start": 0.0}
    return Model(SPIKE_GENERATOR, params)


def _lgn_relay_cell() -> Model:
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


def _lgn_interneuron_cell() -> Model:
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


def _cortex_excitatory_cell() -> Model:
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


def _cortex_inhibitory_cell() -> Model:
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
def _thalamo_noise() -> Model:
    params = {'mean': 0.0, 'std': 1.0}
    return Model(NOISE_GENERATOR, params)
