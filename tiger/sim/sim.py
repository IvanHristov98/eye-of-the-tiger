import time
from typing import List

import numpy as np

import tiger.net as net


class NetRunner:
    _config: net.Config
    _sim_time: float
    _times: List[float]
    _seeds: List[int]
    
    def __init__(self, sim_time: float) -> None:
        self._config = net.Config()
        self._sim_time = sim_time
        self._set_timings()
        self._set_seeds()
    
    def _set_timings(self) -> None:
        times_count = self._sim_time / self._config._sim_step_ms
        self._times = np.zeros(times_count)
        
        for i in np.arange(0, int(times_count)):
            self._times[i] = i * self._config._sim_step_ms

    def _set_seeds(self) -> None:
        np.random.seed(int(time.time()))
        self._seeds = np.arange(self._config.with_nest_threads) + int((time.time()*100)%2**32)
