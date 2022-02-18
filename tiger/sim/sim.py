import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import nest.topology as tp
import nest

import tiger.net.cfg as netcfg
import tiger.net.system as netsys
import tiger.net.layer as lyr


_MULTIMETER_NODE = 'multimeter_node'
_SPIKE_DETECTOR_NODE = 'spike_dector_node'


class NetRunner:
    _config: netcfg.Config
    _sim_time: float
    _times: List[float]
    _seeds: List[int]
    _layers_to_gids: Dict[str, int]
    _layer_ids: List[Tuple[str, int, str]]
    
    def __init__(self, sim_time: float) -> None:
        self._config = netcfg.Config()
        self._sim_time = sim_time
        self._set_timings()
        self._set_seeds()

    def build_network(self) -> None:
        self._set_up_nest()
        
        models, layers, conns = netsys.get_network(self._config)
        
        self._create_models(models)
        self._layer_ids, self._layers_to_gids = self._create_layers(layers)
        self._connect_layers(conns)

    def init_spike_generators(self, retina_spikes: List) -> None:
        midget_ganglion_cells_l_on_spikes = retina_spikes[0]
        midget_ganglion_cells_l_off_spikes = retina_spikes[1]
        midget_ganglion_cells_m_on_spikes = retina_spikes[2]
        midget_ganglion_cells_m_off_spikes = retina_spikes[3]
        
        midget_ganglion_cells_l_on_gid = self._layers_to_gids[lyr.MIDGET_GANGLION_CELLS_L_ON]
        midget_ganglion_cells_l_off_gid = self._layers_to_gids[lyr.MIDGET_GANGLION_CELLS_L_OFF]
        midget_ganglion_cells_m_on_gid = self._layers_to_gids[lyr.MIDGET_GANGLION_CELLS_M_ON]
        midget_ganglion_cells_m_off_gid = self._layers_to_gids[lyr.MIDGET_GANGLION_CELLS_M_OFF]
        
        cell_cnt = 0
        
        for i in np.arange(self._config.lgn_cnt):
            for j in np.arange(self._config.lgn_cnt):
                l_on_cells = tp.GetElement(midget_ganglion_cells_l_on_gid, (i, j))
                nest.SetStatus([l_on_cells[0]], [{'spike_times':midget_ganglion_cells_l_on_spikes[cell_cnt],'spike_weights':[]}])
                
                l_off_cells = tp.GetElement(midget_ganglion_cells_l_off_gid, (i, j))
                nest.SetStatus([l_off_cells[0]], [{'spike_times':midget_ganglion_cells_l_off_spikes[cell_cnt],'spike_weights':[]}])
                
                m_on_cells = tp.GetElement(midget_ganglion_cells_m_on_gid, (i, j))
                nest.SetStatus([m_on_cells[0]], [{'spike_times':midget_ganglion_cells_m_on_spikes[cell_cnt],'spike_weights':[]}])
                
                m_off_cells = tp.GetElement(midget_ganglion_cells_m_off_gid, (i, j))
                nest.SetStatus([m_off_cells[0]], [{'spike_times':midget_ganglion_cells_m_off_spikes[cell_cnt],'spike_weights':[]}])
                
                cell_cnt += 1

    def simulate_with_recording(self, multimeter_models: List, spike_models: List) -> Tuple[List, List]:
        recorders = self._make_recorders(multimeter_models)
        detectors = self._make_spike_detectors(spike_models)
        
        nest.SetStatus([0], {'print_time': True})
        nest.Simulate(self._sim_time)
        
        return recorders, detectors

    def _set_timings(self) -> None:
        times_count = int(self._sim_time / self._config.sim_step_ms)
        self._times = np.zeros(times_count)
        
        for i in np.arange(0, int(times_count)):
            self._times[i] = i * self._config.sim_step_ms

    def _set_seeds(self) -> None:
        np.random.seed(int(time.time()))
        self._seeds = np.arange(self._config.nest_thread_cnt) + int((time.time()*100)%2**32)

    def _set_up_nest(self) -> None:
        nest.ResetKernel()
        nest.ResetNetwork()
        
        nest_kernel_status = {
            "local_num_threads": self._config.nest_thread_cnt,
            "resolution": self._config.sim_step_ms,
            "rng_seeds": list(self._seeds)
        }
        nest.SetKernelStatus(nest_kernel_status)

    def _create_models(self, models: List[Tuple[str, str, Dict]]) -> None:
        for model in models:
            nest.CopyModel(model[0], model[1], model[2])

    def _create_layers(self, layers: List[Tuple[str, Dict]]) -> Tuple[List[Tuple[str, int, str]], Dict[str, Iterable]]:
        layer_ids = []
        layers_to_gids = {}
        
        for layer in layers:
            
            gid = tp.CreateLayer(layer[1])
            
            # layer, gid, cell_type
            layer_ids.append((layer[0], gid[0], layer[1]['elements']))
            layers_to_gids[layer[0]] = gid
            
        return layer_ids, layers_to_gids
  
    def _connect_layers(self, conns: List) -> None:
        for conn in conns:
            src_layer_gids = self._layers_to_gids[conn[0]]
            target_layer_gids = self._layers_to_gids[conn[1]]
            
            tp.ConnectLayers(src_layer_gids, target_layer_gids, conn[2])

    def _make_recorders(self, recorded_models: List) -> List:
        recorder_params = {
            'interval'   : self._config.sim_step_ms,
            'record_from': ['V_m'],
            'record_to'  : ['memory'],
            'withgid'    : True,
            'withtime'   : False
        }
        
        nest.CopyModel('multimeter', _MULTIMETER_NODE, recorder_params)
        
        recorders = []
        
        for pop, model in recorded_models:
            rec = nest.Create(_MULTIMETER_NODE)
            recorders.append([rec, pop, model])
            targets = []
            
            for node in nest.GetLeaves(pop)[0]:
                if nest.GetStatus([node], 'model')[0] == model:
                    targets.append(node)

            nest.Connect(rec, targets)
        
        return recorders

    def _make_spike_detectors(self, recorded_models: List) -> List:
        dector_params = {"withtime": True, "withgid": True, "to_file": False}
        
        nest.CopyModel('spike_detector', _SPIKE_DETECTOR_NODE, dector_params)
        
        detectors = []
        
        for pop, model in recorded_models:
            rec = nest.Create(_SPIKE_DETECTOR_NODE)
            detectors.append([rec, pop, model])
            
            targets = []
            
            for node in nest.GetLeaves(pop)[0]:
                if nest.GetStatus([node], 'model')[0] == model:
                    targets.append(node)
            
            nest.Connect(targets, rec)
