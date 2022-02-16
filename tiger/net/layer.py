from typing import Dict, Tuple

from tiger.net.cfg import Config
import tiger.net.model as mdl


class PopSize:
    rows_color_luminance_exc: int
    rows_luminance_preferring_exc: int
    rows_color_preferring_exc: int
    
    rows_color_luminance_inh: int
    rows_luminance_preferring_inh: int
    rows_color_preferring_inh: int
    
    def __init__(self) -> None:
        pass


def layers(cfg:Config) -> Tuple[str, Dict]:
    ls = [_lgn_layers(cfg)]
    ls += [_cortex_color_luminance_exc_layers(cfg), _cortex_luminance_preferring_exc_layers(cfg), _cortex_color_preferring_exc_layers(cfg)]
    ls += [_cortex_color_luminance_inh_layers(cfg), _cortex_luminance_preferring_inh_layers(cfg), _cortex_color_preferring_inh_layers(cfg)]
    ls += [_noise_gen_layers(cfg)]
    
    return ls


def _lgn_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_lgn_layer_props(cfg)
    
    midget_ganglion_cell_layers = [
        ('Midget_ganglion_cells_L_ON', _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        ('Midget_ganglion_cells_L_OFF', _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        ('Midget_ganglion_cells_M_ON', _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        ('Midget_ganglion_cells_M_OFF', _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
    ]
    parvo_lgn_relay_cell_layers = [
        ('Parvo_LGN_relay_cell_L_ON', _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        ('Parvo_LGN_relay_cell_L_OFF', _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        ('Parvo_LGN_relay_cell_M_ON', _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        ('Parvo_LGN_relay_cell_M_OFF', _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
    ]
    parvo_lgn_interneuron_layers = [
        ('Parvo_LGN_interneuron_ON', _merged_dicts(base_props, {'elements': mdl.LGN_INTERNEURON})),
        ('Parvo_LGN_interneuron_OFF',_merged_dicts(base_props, {'elements': mdl.LGN_INTERNEURON})),
    ]
    
    return midget_ganglion_cell_layers + parvo_lgn_relay_cell_layers + parvo_lgn_interneuron_layers


def _cortex_color_luminance_exc_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_EXC_CELL,
        'rows': pop_size.rows_color_luminance_exc,
        'columns': pop_size.rows_color_luminance_exc
    }
    
    return [
        ('Color_Luminance_L_ON_L_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_L_ON_L_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_L_OFF_L_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_L_OFF_L_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),

        ('Color_Luminance_M_ON_M_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_M_ON_M_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_M_OFF_M_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_M_OFF_M_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),
    ]


def _cortex_luminance_preferring_exc_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_EXC_CELL,
        'rows': pop_size.rows_luminance_preferring_exc,
        'columns': pop_size.rows_luminance_preferring_exc
    }
    
    return [
        ('Luminance_preferring_ON_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_ON_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_OFF_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_OFF_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),
    ]


def _cortex_color_preferring_exc_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_EXC_CELL,
        'rows': pop_size.rows_color_preferring_exc,
        'columns': pop_size.rows_color_preferring_exc,
    }
    
    return [
        ('Color_preferring_L_ON_M_OFF', _merged_dicts(base_props, specific_props.copy())),
        ('Color_preferring_M_ON_L_OFF', _merged_dicts(base_props, specific_props.copy())),
    ]


def _cortex_color_luminance_inh_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_INH_CELL,
        'rows': pop_size.rows_color_luminance_inh,
        'columns': pop_size.rows_color_luminance_inh,
    }
    
    return [
        ('Color_Luminance_inh_L_ON_L_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_L_ON_L_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_L_OFF_L_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_L_OFF_L_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_M_ON_M_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_M_ON_M_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_M_OFF_M_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Color_Luminance_inh_M_OFF_M_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),
    ]


def _cortex_luminance_preferring_inh_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_INH_CELL,
        'rows': pop_size.rows_luminance_preferring_inh,
        'columns': pop_size.rows_luminance_preferring_inh,
    }
    
    return [
        ('Luminance_preferring_inh_ON_OFF_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_inh_ON_OFF_horizontal', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_inh_OFF_ON_vertical', _merged_dicts(base_props, specific_props.copy())),
        ('Luminance_preferring_inh_OFF_ON_horizontal', _merged_dicts(base_props, specific_props.copy())),
    ]


def _cortex_color_preferring_inh_layers(cfg: Config) -> Tuple[str, Dict]:
    base_props = _base_cortex_layer_props()
    pop_size = _pop_size_from_cfg(cfg)
    specific_props =  {
        'elements': mdl.CORTEX_INH_CELL,
        'rows': pop_size.rows_color_preferring_inh,
        'columns': pop_size.rows_color_preferring_inh,
    }
    
    return [
        ('Color_preferring_inh_L_ON_M_OFF', _merged_dicts(base_props, specific_props.copy())),
        ('Color_preferring_inh_M_ON_L_OFF', _merged_dicts(base_props, specific_props.copy())),
    ]


def _noise_gen_layers(cfg: Config) -> Tuple[str, Dict]:
    base_lgn_props = _base_lgn_layer_props()
    lgn_noise_layers = [('Noise_generators_LGN', _merged_dicts(base_lgn_props, {'elements': mdl.THALAMO_NOISE}))]
    
    base_cortex_props = _base_cortex_layer_props()
    ps = _pop_size_from_cfg(cfg)
    
    cortex_exc_noise_layers = [
        ('Noise_generators_Color_Luminance', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_luminance_exc,'columns': ps.rows_color_luminance_exc})),
        ('Noise_generators_Luminance_preferring', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_luminance_preferring_exc,'columns': ps.rows_luminance_preferring_exc})),
        ('Noise_generators_Color_preferring', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_preferring_exc,'columns': ps.rows_color_preferring_exc})),
    ]
    
    cortex_inh_noise_layers = [
        ('Noise_generators_Color_Luminance_inh', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_luminance_inh,'columns': ps.rows_color_luminance_inh})),
        ('Noise_generators_Luminance_preferring_inh', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_luminance_preferring_inh,'columns': ps.rows_luminance_preferring_inh})),
        ('Noise_generators_Color_preferring_inh', _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_preferring_inh,'columns': ps.rows_color_preferring_inh})),
    ]
    
    return lgn_noise_layers + cortex_exc_noise_layers + cortex_inh_noise_layers


def _merged_dicts(a: Dict, b: Dict) -> Dict:
    c = a.copy()
    c.update(b)
    return c


def _pop_size_from_cfg(cfg: Config) -> PopSize:
    pop_size = PopSize()
    pop_size.rows_color_luminance_exc = cfg.cortex_cnt // 2
    pop_size.rows_luminance_preferring_exc = cfg.cortex_cnt
    pop_size.rows_color_preferring_exc = cfg.cortex_cnt // 2

    pop_size.rows_color_luminance_inh = cfg.cortex_cnt // 4
    pop_size.rows_luminance_preferring_inh = cfg.cortex_cnt // 2
    pop_size.rows_color_preferring_inh = cfg.cortex_cnt // 4
    
    return pop_size


def _base_lgn_layer_props(cfg: Config) -> Dict:
    return {
        'rows'     : cfg.lgn_cnt,
        'columns'  : cfg.lgn_cnt,
        'extent'   : [cfg.vis_angle_deg, cfg.vis_angle_deg],
        'edge_wrap': True
    }


def _base_cortex_layer_props(cfg: Config) -> Dict:
    return {
        'rows'     : cfg.cortex_cnt,
        'columns'  : cfg.cortex_cnt,
        'extent'   : [cfg.vis_angle_deg, cfg.vis_angle_deg],
        'edge_wrap': True
    }
