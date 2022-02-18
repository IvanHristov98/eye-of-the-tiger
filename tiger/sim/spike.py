from pathlib import Path
from typing import List

import numpy as np


# def load_spikes(count, ids, folder, stim, trial, layer_sizes, path: Path) -> List:
#     all_spikes = []
#     k = 0

#     for id in ids:
#         spike_times = []
#         lines = [line.rstrip('\n') for line in open(str(path) + "/spikes/" + folder + "/" + str(trial) + "/" + id + stim + ".spikes", "r")]

#         if(isinstance(layer_sizes, list) == False):
#             new_n = count
#         else:
#             new_n = np.sqrt(layer_sizes[k])

#         for n in np.arange(new_n*new_n):
#             h = lines[int(n)].split(',')
#             e = []

#             for element in h[0:len(h)-1]:
#                 e.append(float(element))

#             spike_times.append(e)

#         all_spikes.append(spike_times)
#         k+=1

#     return all_spikes

def gen_spikes(lgn_cnt: int, retina_layers: List[str]) -> List:
    spikes = []
    
    for _ in retina_layers:
        spikes.append([])

    cnt = 0
    
    for _ in np.arange(lgn_cnt):
        for _ in np.arange(lgn_cnt):
            for i in range(len(retina_layers)):
                spikes[i].append([float(cnt + 1)])
        
            cnt += 1

    return spikes
