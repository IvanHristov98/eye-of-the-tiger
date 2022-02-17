from typing import Any, Dict, List, Tuple

import nest
import nest.topology as tp
import numpy as np

from tiger.net.cfg import Config
import tiger.net.layer as lyr
import tiger.net.model as mdl


D = 0.1


class FigConnConfig:
    cfg: Config
    mask_radius_deg: float
    mask_points: List[float]
    center_weight_ns: float
    sigma_deg: float
    mean_delay_ms: float
    std_delay_ms: float
    sources: str
    targets: str
    src_row_cnt: int
    target_row_cnt: int
    
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.mask_points = []


# Returns connections between layers.
def get_connections(cfg: Config):
    pop_size = lyr.pop_size_from_cfg()
    _copy_synapse_model()

    # LGN connections    
    conns = _retinal_ganglion_cells_to_relay_cells(cfg)
    conns += _retinal_ganglion_cells_to_interneurons(cfg)
    conns += _interneurons_to_relay_cells(cfg)
    conns += _interneurons_to_interneurons(cfg)
    
    # Thalamocortical connections
    conns += _relay_cells_to_color_luminance_cells(cfg, pop_size)
    
    return conns


def _copy_synapse_model() -> Any:
    nest.CopyModel(mdl.STATIC_SYNAPSE, mdl.SYN)


def _retinal_ganglion_cells_to_relay_cells(cfg: Config) -> List:
    fig_cfg = FigConnConfig(cfg)
    fig_cfg.center_weight_ns = 4.0
    fig_cfg.sigma_deg = 0.03
    fig_cfg.mean_delay_ms = 3.0
    fig_cfg.std_delay_ms = 1.0
    fig_cfg.mask_radius_deg = fig_cfg.sigma_deg * 3.0
    fig_cfg.sources = mdl.RETINAL_GANGLION_CELL
    fig_cfg.targets = mdl.LGN_RELAY_CELL
    fig_cfg.src_row_cnt = cfg.lgn_cnt
    fig_cfg.target_row_cnt = cfg.lgn_cnt
    
    retina_to_relay_params = _make_circular_conn_dict(fig_cfg)
    
    return [
        [lyr.MIDGET_GANGLION_CELLS_L_ON, lyr.PARVO_LGN_RELAY_CELL_L_ON, retina_to_relay_params],
        [lyr.MIDGET_GANGLION_CELLS_L_OFF, lyr.PARVO_LGN_RELAY_CELL_L_OFF, retina_to_relay_params],
        [lyr.MIDGET_GANGLION_CELLS_M_ON, lyr.PARVO_LGN_RELAY_CELL_M_ON, retina_to_relay_params],
        [lyr.MIDGET_GANGLION_CELLS_M_OFF, lyr.PARVO_LGN_RELAY_CELL_M_OFF, retina_to_relay_params],
    ]


def _retinal_ganglion_cells_to_interneurons(cfg: Config) -> List:
    fig_cfg = FigConnConfig(cfg)
    fig_cfg.center_weight_ns = 2.0
    fig_cfg.sigma_deg = 0.06
    fig_cfg.mean_delay_ms = 3.0
    fig_cfg.std_delay_ms = 1.0
    fig_cfg.mask_radius_deg = fig_cfg.sigma_deg * 3.0
    fig_cfg.sources = mdl.RETINAL_GANGLION_CELL
    fig_cfg.targets = mdl.LGN_INTERNEURON
    fig_cfg.src_row_cnt = cfg.lgn_cnt
    fig_cfg.target_row_cnt = cfg.lgn_cnt
    
    retina_to_interneuron_params = _make_circular_conn_dict(fig_cfg)
    
    return [
        [lyr.MIDGET_GANGLION_CELLS_L_ON, lyr.PARVO_LGN_INTERNEURON_ON, retina_to_interneuron_params],
        [lyr.MIDGET_GANGLION_CELLS_M_ON, lyr.PARVO_LGN_INTERNEURON_ON, retina_to_interneuron_params],
        [lyr.MIDGET_GANGLION_CELLS_L_OFF, lyr.PARVO_LGN_INTERNEURON_OFF, retina_to_interneuron_params],
        [lyr.MIDGET_GANGLION_CELLS_M_OFF, lyr.PARVO_LGN_INTERNEURON_OFF, retina_to_interneuron_params],
    ]


# inhibitory synapses
def _interneurons_to_relay_cells(cfg: Config) -> List:
    fig_cfg = FigConnConfig(cfg)
    fig_cfg.center_weight_ns = -2.0
    fig_cfg.sigma_deg = 0.06
    fig_cfg.mean_delay_ms = 2.0 # (Hill 2005)
    fig_cfg.std_delay_ms = 0.25 # (Hill 2005)
    fig_cfg.mask_radius_deg = fig_cfg.sigma_deg * 3.0
    fig_cfg.sources = mdl.LGN_INTERNEURON
    fig_cfg.targets = mdl.LGN_RELAY_CELL
    fig_cfg.src_row_cnt = cfg.lgn_cnt
    fig_cfg.target_row_cnt = cfg.lgn_cnt
    
    interneuron_to_relay_params = _make_circular_conn_dict(fig_cfg)
    
    return [
        [lyr.PARVO_LGN_INTERNEURON_ON, lyr.PARVO_LGN_RELAY_CELL_L_ON, interneuron_to_relay_params],
        [lyr.PARVO_LGN_INTERNEURON_ON, lyr.PARVO_LGN_RELAY_CELL_M_ON, interneuron_to_relay_params],
        [lyr.PARVO_LGN_INTERNEURON_OFF, lyr.PARVO_LGN_RELAY_CELL_L_OFF, interneuron_to_relay_params],
        [lyr.PARVO_LGN_INTERNEURON_OFF, lyr.PARVO_LGN_RELAY_CELL_M_OFF, interneuron_to_relay_params],
    ]


# inhibitory synapses
def _interneurons_to_interneurons(cfg: Config) -> List:
    fig_cfg = FigConnConfig(cfg)
    fig_cfg.center_weight_ns = -2.0
    fig_cfg.sigma_deg = 0.06
    fig_cfg.mean_delay_ms = 2.0
    fig_cfg.std_delay_ms = 0.25
    fig_cfg.mask_radius_deg = fig_cfg.sigma_deg * 3.0
    fig_cfg.sources = mdl.LGN_INTERNEURON
    fig_cfg.targets = mdl.LGN_INTERNEURON
    fig_cfg.src_row_cnt = cfg.lgn_cnt
    fig_cfg.target_row_cnt = cfg.lgn_cnt
    
    interneuron_to_interneuron_params = _make_circular_conn_dict(fig_cfg)
    
    return [
        [lyr.PARVO_LGN_INTERNEURON_ON, lyr.PARVO_LGN_INTERNEURON_ON, interneuron_to_interneuron_params],
        [lyr.PARVO_LGN_INTERNEURON_OFF, lyr.PARVO_LGN_INTERNEURON_OFF, interneuron_to_interneuron_params],
    ]


def _relay_cells_to_color_luminance_cells(cfg: Config, pop_size: lyr.PopSize) -> List:
    fig_cfg = FigConnConfig(cfg)
    fig_cfg.center_weight_ns = 2.5
    fig_cfg.mean_delay_ms = 3.0 # (Hill_2005)
    fig_cfg.std_delay_ms = 0.25 # (Hill_2005)
    fig_cfg.sources = mdl.LGN_RELAY_CELL
    fig_cfg.target_row_cnt = mdl.CORTEX_EXC_CELL
    fig_cfg.src_row_cnt = cfg.lgn_cnt
    fig_cfg.target_row_cnt = pop_size.rows_color_luminance_exc
    
    fig_cfg.mask_points = [-0.026, -0.06, 0.026, 0.06]
    color_luminance_vertical_on_params = _make_rect_conn_dict(fig_cfg)
    
    fig_cfg.mask_points = [-0.026 + D, -0.06, 0.026 + D, 0.06]
    color_luminance_vertical_off_params = _make_rect_conn_dict(fig_cfg)
    
    fig_cfg.mask_points = [-0.06, -0.026, 0.06, 0.026]
    color_luminance_horizontal_on_params = _make_rect_conn_dict(fig_cfg)
    
    fig_cfg.mask_points = [-0.06, -0.026 + D, 0.06, 0.026 + D]
    color_luminance_horizontal_off_params = _make_rect_conn_dict(fig_cfg)
    
    return [
        # L_ON_L_OFF
        [lyr.PARVO_LGN_RELAY_CELL_L_ON, lyr.COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL, color_luminance_vertical_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_OFF, lyr.COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL, color_luminance_vertical_off_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_ON, lyr.COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL, color_luminance_horizontal_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_OFF, lyr.COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL, color_luminance_horizontal_off_params],
        # L_OFF_L_ON
        [lyr.PARVO_LGN_RELAY_CELL_L_OFF, lyr.COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL, color_luminance_vertical_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_ON, lyr.COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL, color_luminance_vertical_off_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_OFF, lyr.COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL, color_luminance_horizontal_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_L_ON, lyr.COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL, color_luminance_horizontal_off_params],
        # M_ON_M_OFF
        [lyr.PARVO_LGN_RELAY_CELL_M_ON, lyr.COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL, color_luminance_vertical_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_OFF, lyr.COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL, color_luminance_vertical_off_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_ON, lyr.COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL, color_luminance_horizontal_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_OFF, lyr.COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL, color_luminance_horizontal_off_params],
        # M_OFF_M_ON
        [lyr.PARVO_LGN_RELAY_CELL_M_OFF, lyr.COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL, color_luminance_vertical_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_ON, lyr.COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL, color_luminance_vertical_off_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_OFF, lyr.COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL, color_luminance_horizontal_on_params],
        [lyr.PARVO_LGN_RELAY_CELL_M_ON, lyr.COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL, color_luminance_horizontal_off_params],
    ]


def _make_circular_conn_dict(cfg: FigConnConfig, kernel=1.0) -> Dict:
    total_weight, _ = _get_relative_weight_for_circular_mask(cfg)
    
    return {
        "connection_type":"divergent",
        "mask": {"circular": {"radius": cfg.mask_radius_deg}},
        "kernel": kernel,
        "delays" : {"normal": {"mean": cfg.mean_delay_ms, "std": cfg.std_delay_ms, "min": cfg.cfg.sim_step_ms}},
        "synapse_model": mdl.SYN,
        "weights": {"gaussian": {"p_center": cfg.center_weight_ns/total_weight, "sigma_deg": cfg.sigma_deg}},
        "sources": {"model": cfg.sources},
        "targets": {"model": cfg.targets},
        "allow_autapses": False,
        "allow_multapses": False
    }


def _get_relative_weight_for_circular_mask(cfg: FigConnConfig) -> Tuple[float, int]:
    # Create a fictional network and count the number of target connections
    src_layer = _create_src_layer(cfg)
    target_layer = _create_target_layer(cfg)

    # Circular mask
    mask_params = {
        "connection_type":"divergent",
        "mask": {"circular": {"radius": cfg.mask_radius_deg}},
        "kernel": 1.0,
        "weights": {"gaussian": {"p_center": 1.0, "sigma_deg": cfg.mask_radius_deg/3.0}}
    }
    
    return _get_relative_weight(src_layer, target_layer, mask_params)


def _make_rect_conn_dict(cfg: FigConnConfig) -> Dict:
    total_weight, _ = _get_relative_weight_for_rect_mask(cfg)
    
    return {
        "connection_type":"convergent",
        "mask": {'rectangular': {'lower_left':[cfg.mask_points[0], cfg.mask_points[1]],
                                'upper_right':[cfg.mask_points[2], cfg.mask_points[3]]}},
        "kernel": 1.0,
        "delays": {"normal": {"mean": cfg.mean_delay_ms, "std": cfg.std_delay_ms, "min": cfg.cfg.sim_step_ms}},
        "synapse_model": mdl.SYN,
        "weights": cfg.center_weight_ns/total_weight,
        "sources": {"model": cfg.sources},
        "targets": {"model": cfg.targets},
        "allow_autapses": False,
        "allow_multapses": False
    }


def _get_relative_weight_for_rect_mask(cfg: FigConnConfig) -> Tuple[float, int]:
    # Create a fictional network and count the number of target connections
    src_layer = _create_src_layer(cfg)
    target_layer = _create_target_layer(cfg)

    # Rectangular mask
    mask_params = {
        "connection_type":"convergent",
        "mask": {'rectangular': {'lower_left':[cfg.mask_points[0], cfg.mask_points[1]],
                                'upper_right':[cfg.mask_points[2], cfg.mask_points[3]]}},
        "kernel": 1.0,
        "weights": 1.0
    }
    
    return _get_relative_weight(src_layer, target_layer, mask_params)


# Weights are normalized with the size of the network so that the sum of the weights
# of all incoming synapses is always equal to a constant value.
def _get_relative_weight(src_layer: Any, target_layer: Any, mask_params: Dict) -> Tuple[float, int]:
    tp.ConnectLayers(src_layer,target_layer, mask_params)
    ctr = tp.FindCenterElement(target_layer)
    conn = nest.GetConnections(target=[ctr[0]])

    st = nest.GetStatus(conn)

    w = 0.0
    for n in np.arange(len(st)):
        w += st[n]['weight']

    if w == 0.0:
        print ("Warning: found w = 0.0. Changed to 1.0.")
        w = 1.0

    return w, len(conn)


def _create_src_layer(cfg: FigConnConfig) -> Any:
    layer_props = {
    'rows'     : cfg.src_row_cnt,
    'columns'  : cfg.src_row_cnt,
    'extent'   : [cfg.cfg.vis_angle_deg, cfg.cfg.vis_angle_deg],
    'edge_wrap': True,
    'elements': 'iaf_cond_exp'
    }

    return tp.CreateLayer(layer_props)


def _create_target_layer(cfg: FigConnConfig) -> Any:
    layerProps = {
        'rows'     : cfg.target_row_cnt,
        'columns'  : cfg.target_row_cnt,
        'extent'   : [cfg.cfg.vis_angle_deg, cfg.cfg.vis_angle_deg],
        'edge_wrap': True,
        'elements': 'iaf_cond_exp'
    }

    return tp.CreateLayer(layerProps)
