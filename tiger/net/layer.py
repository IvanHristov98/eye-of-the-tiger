from typing import Dict, Tuple

from tiger.net.cfg import Config
import tiger.net.model as mdl


MIDGET_GANGLION_CELLS_L_ON = "Midget_ganglion_cells_L_ON"
MIDGET_GANGLION_CELLS_L_OFF = "Midget_ganglion_cells_L_OFF"
MIDGET_GANGLION_CELLS_M_ON = "Midget_ganglion_cells_M_ON"
MIDGET_GANGLION_CELLS_M_OFF = "Midget_ganglion_cells_M_OFF"

PARVO_LGN_RELAY_CELL_L_ON = "Parvo_LGN_relay_cell_L_ON"
PARVO_LGN_RELAY_CELL_L_OFF = "Parvo_LGN_relay_cell_L_OFF"
PARVO_LGN_RELAY_CELL_M_ON = "Parvo_LGN_relay_cell_M_ON"
PARVO_LGN_RELAY_CELL_M_OFF = "Parvo_LGN_relay_cell_M_OFF"
PARVO_LGN_INTERNEURON_ON = "Parvo_LGN_interneuron_ON"
PARVO_LGN_INTERNEURON_OFF = "Parvo_LGN_interneuron_OFF"

COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL = "Color_Luminance_L_ON_L_OFF_vertical"
COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL = "Color_Luminance_L_ON_L_OFF_horizontal"
COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL = "Color_Luminance_L_OFF_L_ON_vertical"
COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL = "Color_Luminance_L_OFF_L_ON_horizontal"
COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL = "Color_Luminance_M_ON_M_OFF_vertical"
COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL = "Color_Luminance_M_ON_M_OFF_horizontal"
COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL = "Color_Luminance_M_OFF_M_ON_vertical"
COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL = "Color_Luminance_M_OFF_M_ON_horizontal"

LUMINANCE_PREFERRING_ON_OFF_VERTICAL = "Luminance_preferring_ON_OFF_vertical"
LUMINANCE_PREFERRING_ON_OFF_HORIZONTAL = "Luminance_preferring_ON_OFF_horizontal"
LUMINANCE_PREFERRING_OFF_ON_VERTICAL = "Luminance_preferring_OFF_ON_vertical"
LUMINANCE_PREFERRING_OFF_ON_HORIZONTAL = "Luminance_preferring_OFF_ON_horizontal"

COLOR_PREFERRING_L_ON_M_OFF = "Color_preferring_L_ON_M_OFF"
COLOR_PREFERRING_M_ON_L_OFF = "Color_preferring_M_ON_L_OFF"

COLOR_LUMINANCE_INH_L_ON_L_OFF_VERTICAL = "Color_Luminance_inh_L_ON_L_OFF_vertical"
COLOR_LUMINANCE_INH_L_ON_L_OFF_HORIZONTAL = "Color_Luminance_inh_L_ON_L_OFF_horizontal"
COLOR_LUMINANCE_INH_L_OFF_L_ON_VERTICAL = "Color_Luminance_inh_L_OFF_L_ON_vertical"
COLOR_LUMINANCE_INH_L_OFF_L_ON_HORIZONTAL = "Color_Luminance_inh_L_OFF_L_ON_horizontal"
COLOR_LUMINANCE_INH_M_ON_M_OFF_VERTICAL = "Color_Luminance_inh_M_ON_M_OFF_vertical"
COLOR_LUMINANCE_INH_M_ON_M_OFF_HORIZONTAL = "Color_Luminance_inh_M_ON_M_OFF_horizontal"
COLOR_LUMINANCE_INH_M_OFF_M_ON_VERTICAL = "Color_Luminance_inh_M_OFF_M_ON_vertical"
COLOR_LUMINANCE_INH_M_OFF_M_ON_HORIZONTAL = "Color_Luminance_inh_M_OFF_M_ON_horizontal"

LUMINANCE_PREFERRING_INH_ON_OFF_VERTICAL = "Luminance_preferring_inh_ON_OFF_vertical"
LUMINANCE_PREFERRING_INH_ON_OFF_HORIZONTAL = "Luminance_preferring_inh_ON_OFF_horizontal"
LUMINANCE_PREFERRING_INH_OFF_ON_VERTICAL = "Luminance_preferring_inh_OFF_ON_vertical"
LUMINANCE_PREFERRING_INH_OFF_ON_HORIZONTAL = "Luminance_preferring_inh_OFF_ON_horizontal"

COLOR_PREFERRING_INH_L_ON_M_OFF = "Color_preferring_inh_L_ON_M_OFF"
COLOR_PREFERRING_INH_M_ON_L_OFF = "Color_preferring_inh_M_ON_L_OFF"

NOISE_GENERATORS_LGN = "Noise_generators_LGN"
NOISE_GENERATORS_COLOR_LUMINANCE = "Noise_generators_Color_Luminance"
NOISE_GENERATORS_LUMINANCE_PREFERRING = "Noise_generators_Luminance_preferring"
NOISE_GENERATORS_COLOR_PREFERRING = "Noise_generators_Color_preferring"
NOISE_GENERATORS_COLOR_LUMINANCE_INH = "Noise_generators_Color_Luminance_inh"
NOISE_GENERATORS_LUMINANCE_PREFERRING_INH = "Noise_generators_Luminance_preferring_inh"
NOISE_GENERATORS_COLOR_PREFERRING_INH = "Noise_generators_Color_preferring_inh"


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
        (MIDGET_GANGLION_CELLS_L_ON, _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        (MIDGET_GANGLION_CELLS_L_OFF, _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        (MIDGET_GANGLION_CELLS_M_ON, _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
        (MIDGET_GANGLION_CELLS_M_OFF, _merged_dicts(base_props, {'elements': mdl.RETINAL_GANGLION_CELL})),
    ]
    parvo_lgn_relay_cell_layers = [
        (PARVO_LGN_RELAY_CELL_L_ON, _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        (PARVO_LGN_RELAY_CELL_L_OFF, _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        (PARVO_LGN_RELAY_CELL_M_ON, _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
        (PARVO_LGN_RELAY_CELL_M_OFF, _merged_dicts(base_props, {'elements': mdl.LGN_RELAY_CELL})),
    ]
    parvo_lgn_interneuron_layers = [
        (PARVO_LGN_INTERNEURON_ON, _merged_dicts(base_props, {'elements': mdl.LGN_INTERNEURON})),
        (PARVO_LGN_INTERNEURON_OFF,_merged_dicts(base_props, {'elements': mdl.LGN_INTERNEURON})),
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
        (COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),

        (COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
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
        (LUMINANCE_PREFERRING_ON_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_ON_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_OFF_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_OFF_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
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
        (COLOR_PREFERRING_L_ON_M_OFF, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_PREFERRING_M_ON_L_OFF, _merged_dicts(base_props, specific_props.copy())),
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
        (COLOR_LUMINANCE_INH_L_ON_L_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_L_ON_L_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_L_OFF_L_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_L_OFF_L_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_M_ON_M_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_M_ON_M_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_M_OFF_M_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_LUMINANCE_INH_M_OFF_M_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
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
        (LUMINANCE_PREFERRING_INH_ON_OFF_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_INH_ON_OFF_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_INH_OFF_ON_VERTICAL, _merged_dicts(base_props, specific_props.copy())),
        (LUMINANCE_PREFERRING_INH_OFF_ON_HORIZONTAL, _merged_dicts(base_props, specific_props.copy())),
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
        (COLOR_PREFERRING_INH_L_ON_M_OFF, _merged_dicts(base_props, specific_props.copy())),
        (COLOR_PREFERRING_INH_M_ON_L_OFF, _merged_dicts(base_props, specific_props.copy())),
    ]


def _noise_gen_layers(cfg: Config) -> Tuple[str, Dict]:
    base_lgn_props = _base_lgn_layer_props()
    lgn_noise_layers = [(NOISE_GENERATORS_LGN, _merged_dicts(base_lgn_props, {'elements': mdl.THALAMO_NOISE}))]
    
    base_cortex_props = _base_cortex_layer_props()
    ps = _pop_size_from_cfg(cfg)
    
    cortex_exc_noise_layers = [
        (NOISE_GENERATORS_COLOR_LUMINANCE, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_luminance_exc,'columns': ps.rows_color_luminance_exc})),
        (NOISE_GENERATORS_LUMINANCE_PREFERRING, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_luminance_preferring_exc,'columns': ps.rows_luminance_preferring_exc})),
        (NOISE_GENERATORS_COLOR_PREFERRING, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_preferring_exc,'columns': ps.rows_color_preferring_exc})),
    ]
    
    cortex_inh_noise_layers = [
        (NOISE_GENERATORS_COLOR_LUMINANCE_INH, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_color_luminance_inh,'columns': ps.rows_color_luminance_inh})),
        (NOISE_GENERATORS_LUMINANCE_PREFERRING_INH, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
            'rows': ps.rows_luminance_preferring_inh,'columns': ps.rows_luminance_preferring_inh})),
        (NOISE_GENERATORS_COLOR_PREFERRING_INH, _merged_dicts(base_cortex_props, {'elements': mdl.THALAMO_NOISE,
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
