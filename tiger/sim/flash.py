#!/usr/bin/env python3

from typing import List, Tuple
import shutil
from pathlib import Path
import os

import nest
import matplotlib.pyplot as plt

import tiger.sim.sim as sim
import tiger.net.layer as lyr
import tiger.sim.spike as sp


DATA_DIR = "DATA_DIR"


class FlashExperiment:
    sim_time: float
    net_runner: sim.NetRunner
    trial_cnt: int
    stimulus_id: str
    plot_start_time: float
    bin_size: float
    layers_to_track: List[str]
    topo_layers: List[str]
    pop_layers: List[str]
    plot_intracellular: bool
    plot_PSTH: bool
    plot_topographical: bool
    intracellular_rows: int
    intracellular_cols: int
    intracellular_starting_row: int
    intracellular_starting_col: int
    layers_to_record: List[int]
    potentials: List
    spikes: List
    spike_subfolder: str
    retina_labels: List[str]
    lgn_count: int
    cortex_cnt: int
    layer_sizes: List[int]
    
    def __init__(self) -> None:
        self.sim_time = 50.0
        self.net_runner = sim.NetRunner(self.sim_time)
        self.trial_cnt = 2
        self.spike_subfolder = "flash"
        self.stimulus_id = "_square_"
        self.plot_start_time = 200.0
        
        # cell to analyze
        selected_cell = []
        # should we use the central cell
        is_center_cell = True
        
        self.lgn_cnt = self.net_runner.config.lgn_cnt
        self.cortex_cnt = self.net_runner.config.cortex_cnt
        
        self.bin_size = 10.0
        
        self.layers_to_track = [lyr.PARVO_LGN_RELAY_CELL_L_ON, lyr.PARVO_LGN_RELAY_CELL_L_OFF, lyr.PARVO_LGN_RELAY_CELL_M_ON, lyr.PARVO_LGN_RELAY_CELL_M_OFF]
        self.layers_to_track += [lyr.PARVO_LGN_INTERNEURON_ON, lyr.PARVO_LGN_INTERNEURON_OFF]
        self.layers_to_track += [
            lyr.COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL,
            lyr.COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL,
        ]
        self.layers_to_track += [
            lyr.LUMINANCE_PREFERRING_ON_OFF_VERTICAL,
            lyr.LUMINANCE_PREFERRING_ON_OFF_HORIZONTAL,
            lyr.LUMINANCE_PREFERRING_OFF_ON_VERTICAL,
            lyr.LUMINANCE_PREFERRING_OFF_ON_HORIZONTAL,
        ]
        self.layers_to_track += [
            lyr.COLOR_PREFERRING_L_ON_M_OFF,
            lyr.COLOR_PREFERRING_M_ON_L_OFF,
        ]
        
        self.layers_to_track += [
            lyr.COLOR_LUMINANCE_INH_L_ON_L_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_L_ON_L_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_L_OFF_L_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_L_OFF_L_ON_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_M_ON_M_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_M_ON_M_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_M_OFF_M_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_M_OFF_M_ON_HORIZONTAL,
        ]
        self.layers_to_track += [
            lyr.LUMINANCE_PREFERRING_INH_ON_OFF_VERTICAL,
            lyr.LUMINANCE_PREFERRING_INH_ON_OFF_HORIZONTAL,
            lyr.LUMINANCE_PREFERRING_INH_OFF_ON_VERTICAL,
            lyr.LUMINANCE_PREFERRING_INH_OFF_ON_HORIZONTAL,
        ]
        self.layers_to_track += [
            lyr.COLOR_PREFERRING_INH_L_ON_M_OFF,
            lyr.COLOR_PREFERRING_INH_M_ON_L_OFF,
        ]
        
        self.topo_layers = [
            lyr.PARVO_LGN_RELAY_CELL_L_ON,
            lyr.PARVO_LGN_RELAY_CELL_L_OFF,
            
            lyr.COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL,
            lyr.LUMINANCE_PREFERRING_ON_OFF_VERTICAL,
            lyr.COLOR_PREFERRING_L_ON_M_OFF,
            
            lyr.COLOR_LUMINANCE_INH_L_ON_L_OFF_VERTICAL,
            lyr.LUMINANCE_PREFERRING_INH_ON_OFF_VERTICAL,
            lyr.COLOR_PREFERRING_INH_L_ON_M_OFF,
        ]
        
        self.pop_layers = [
            lyr.COLOR_LUMINANCE_L_ON_L_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_L_ON_L_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_L_OFF_L_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_L_OFF_L_ON_HORIZONTAL,
            lyr.COLOR_LUMINANCE_M_ON_M_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_M_ON_M_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_M_OFF_M_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_M_OFF_M_ON_HORIZONTAL,
            
            lyr.LUMINANCE_PREFERRING_ON_OFF_VERTICAL,
            lyr.LUMINANCE_PREFERRING_ON_OFF_HORIZONTAL,
            lyr.LUMINANCE_PREFERRING_OFF_ON_VERTICAL,
            lyr.LUMINANCE_PREFERRING_OFF_ON_HORIZONTAL,
            
            lyr.COLOR_PREFERRING_L_ON_M_OFF,
            lyr.COLOR_PREFERRING_M_ON_L_OFF,
            
            lyr.COLOR_LUMINANCE_INH_L_ON_L_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_L_ON_L_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_L_OFF_L_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_L_OFF_L_ON_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_M_ON_M_OFF_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_M_ON_M_OFF_HORIZONTAL,
            lyr.COLOR_LUMINANCE_INH_M_OFF_M_ON_VERTICAL,
            lyr.COLOR_LUMINANCE_INH_M_OFF_M_ON_HORIZONTAL,
            
            lyr.LUMINANCE_PREFERRING_INH_ON_OFF_VERTICAL,
            lyr.LUMINANCE_PREFERRING_INH_ON_OFF_HORIZONTAL,
            lyr.LUMINANCE_PREFERRING_INH_OFF_ON_VERTICAL,
            lyr.LUMINANCE_PREFERRING_INH_OFF_ON_HORIZONTAL,
            
            lyr.COLOR_PREFERRING_INH_L_ON_M_OFF,
            lyr.COLOR_PREFERRING_INH_M_ON_L_OFF,
        ]
        
        self.plot_intracellular = False
        self.plot_PSTH = False
        self.plot_topographical = True

        # Individual intracellular traces
        self.intracellular_rows = 4
        self.intracellular_cols = 5
        self.intracellular_starting_row = 0
        self.intracellular_starting_col = 0
        
        self.layers_to_record = []
        
        self.potentials = []
        self.spikes = []

        # Retina references
        self.retina_labels = [
            lyr.MIDGET_GANGLION_CELLS_L_ON,
            lyr.MIDGET_GANGLION_CELLS_L_OFF,
            lyr.MIDGET_GANGLION_CELLS_M_ON,
            lyr.MIDGET_GANGLION_CELLS_M_OFF,
        ]
        
    def init_dirs(self) -> None:
        data_dir = Path(os.environ[DATA_DIR])
        
        if data_dir.exists() and data_dir.is_dir():
            shutil.rmtree(data_dir)
        
        os.mkdir(data_dir)

        res_dir = Path(data_dir, "res")
        os.mkdir(res_dir)
        
        subdata_dir = Path(data_dir, "data")
        os.mkdir(subdata_dir)
        
        spikes_dir = Path(data_dir, "spikes")
        os.mkdir(spikes_dir)
        
        spikes_subdir = Path(spikes_dir, self.spike_subfolder)
        os.mkdir(spikes_subdir)

    def simulate(self) -> None:
        self.net_runner.build_network()
        retina_spikes = sp.gen_spikes(self.lgn_cnt, self.retina_labels)
        self.net_runner.init_spike_generators(retina_spikes)
        
        self._load_layers_to_record(self.net_runner.layer_ids)
        multimeters, detectors = self.net_runner.simulate_with_recording(self.layers_to_record,  self.layers_to_record)
        
        data_dir = Path(os.environ[DATA_DIR])
        
        for multimeter in multimeters:
            data = nest.GetStatus(multimeter[0])[0]['events']
            
            plt.figure(1)
            print(len(data['times']))
            plt.plot(data['times'][:5000], data['V_m'][:5000])
            plt.savefig(str(Path(data_dir, f"{str(multimeter[0][0])}-{str(multimeter[2])}.png")))
            plt.clf()

    def _load_layers_to_record(self, layer_ids: List[Tuple[str, int, str]]) -> None:
        self.layers_to_record = []
        self.layer_sizes = []
        
        for layer in self.layers_to_track:
            found = False
            
            for layer_id in layer_ids:
                if layer_id[0] == layer:
                    self.layers_to_record.append((layer_id[1], layer_id[2]))
                    self.layer_sizes.append(len(nest.GetNodes(layer_id[1])[0]))
                    found = True
            
            if not found:
                print(f"Layer {layer} not found...")
                self.layers_to_record.append((layer_id[1], layer_id[2]))


def main():
    exp = FlashExperiment()
    exp.init_dirs()
    
    print("SIMULATING")
    exp.simulate()
    
    print("DONE SIMULATING")


if __name__ == "__main__":
    main()
