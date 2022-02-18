from typing import Tuple

import tiger.net.cfg as cfg
import tiger.net.model as mdl
import tiger.net.layer as lyr
import tiger.net.conn as conn


def get_network(cfg: cfg.Config) -> Tuple:
    models = mdl.get_models()
    layers = lyr.layers(cfg)
    conns = conn.get_connections(cfg)
    
    return models, layers, conns
